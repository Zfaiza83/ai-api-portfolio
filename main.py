from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="AI Text API",
    description="API de traitement de texte propulsée par Claude",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class TextInput(BaseModel):
    text: str
    language: str = "français"


class TextResponse(BaseModel):
    result: str
    input_length: int
    model: str = "claude-opus-4-5"


def call_claude(prompt: str) -> str:
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "message": "API opérationnelle"}


@app.post("/summarize", response_model=TextResponse)
async def summarize(body: TextInput):
    if len(body.text) < 50:
        raise HTTPException(status_code=400, detail="Texte trop court (minimum 50 caractères)")
    prompt = f"""Résume ce texte en 3 à 5 phrases claires et concises.
Garde les informations essentielles. Ne rajoute aucun commentaire.

Texte :
{body.text}

Résumé :"""
    result = call_claude(prompt)
    return TextResponse(result=result, input_length=len(body.text))


@app.post("/extract-keywords", response_model=TextResponse)
async def extract_keywords(body: TextInput):
    if len(body.text) < 30:
        raise HTTPException(status_code=400, detail="Texte trop court (minimum 30 caractères)")
    prompt = f"""Extrais les 5 à 10 mots-clés ou expressions clés de ce texte.
Retourne-les sous forme de liste simple, un par ligne, sans numérotation ni explication.

Texte :
{body.text}

Mots-clés :"""
    result = call_claude(prompt)
    return TextResponse(result=result, input_length=len(body.text))


@app.post("/translate", response_model=TextResponse)
async def translate(body: TextInput):
    if len(body.text) < 5:
        raise HTTPException(status_code=400, detail="Texte trop court")
    prompt = f"""Traduis ce texte en {body.language}.
Retourne uniquement la traduction, sans commentaire.

Texte original :
{body.text}

Traduction en {body.language} :"""
    result = call_claude(prompt)
    return TextResponse(result=result, input_length=len(body.text))


@app.post("/improve", response_model=TextResponse)
async def improve(body: TextInput):
    if len(body.text) < 20:
        raise HTTPException(status_code=400, detail="Texte trop court (minimum 20 caractères)")
    prompt = f"""Améliore ce texte : corrige les fautes, améliore la fluidité et la clarté.
Garde le sens original. Retourne uniquement le texte amélioré.

Texte :
{body.text}

Texte amélioré :"""
    result = call_claude(prompt)
    return TextResponse(result=result, input_length=len(body.text))
