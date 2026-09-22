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

# Chemin absolu vers le dossier static
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="AI Text API",
    description="API de traitement de texte propulsée par Claude",
    version="1.0.0"
)

# CORS pour autoriser le frontend à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques (frontend)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Client Claude
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# --- Modèles de données ---

class TextInput(BaseModel):
    text: str
    language: str = "français"  # pour la traduction


class TextResponse(BaseModel):
    result: str
    input_length: int
    model: str = "claude-opus-4-5"


# --- Helper ---

def call_claude(prompt: str) -> str:
    """Appelle Claude et retourne la réponse texte."""
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# --- Endpoints ---

@app.get("/")
async def root():
    """Page d'accueil — sert l'interface HTML."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    """Vérifie que l'API est en ligne."""
    return {"status": "ok", "message": "API opérationnelle"}


@app.post("/summarize", response_model=TextResponse)
async def summarize(body: TextInput):
    """
    Résume un texte en 3-5 phrases.

    - **text** : le texte à résumer (min 50 caractères)
    """
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
    """
    Extrait les 5 à 10 mots-clés principaux d'un texte.

    - **text** : le texte à analyser
    """
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
    """
    Traduit un texte dans la langue cible.

    - **text** : le texte à traduire
    - **language** : langue cible (ex: anglais, espagnol, arabe...)
    """
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
    """
    Améliore la qualité rédactionnelle d'un texte.

    - **text** : le texte à améliorer
    """
    if len(body.text) < 20:
        raise HTTPException(status_code=400, detail="Texte trop court (minimum 20 caractères)")

    prompt = f"""Améliore ce texte : corrige les fautes, améliore la fluidité et la clarté.
Garde le sens original. Retourne uniquement le texte amélioré.

Texte :
{body.text}

Texte amélioré :"""

    result = call_claude(prompt)
    return TextResponse(result=result, input_length=len(body.text))
