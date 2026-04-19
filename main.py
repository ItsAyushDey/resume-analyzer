from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
import PyPDF2
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(".env"))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

api_key = os.getenv("AIzaSyDCQsrEkDdieOT0O19uDPe7-6NOKEvX8lM")
model = None

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is missing. Set GEMINI_API_KEY in .env.",
        )

    try:
        text = extract_text_from_pdf(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text was found in the PDF.")

    prompt = f"""
    Analyze this resume and give:
    1. Score out of 100
    2. Strengths
    3. Weaknesses
    4. Suggestions

    Resume:
    {text}
    """

    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini analysis failed: {exc}") from exc

    return {"analysis": response.text}