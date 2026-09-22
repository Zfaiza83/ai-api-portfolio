from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Text API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Text API</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 2rem 1rem; }
    header { text-align: center; margin-bottom: 2.5rem; }
    header h1 { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    header p { color: #94a3b8; margin-top: 0.5rem; font-size: 0.95rem; }
    .container { max-width: 760px; margin: 0 auto; }
    .card { background: #1e2130; border: 1px solid #2d3148; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
    .tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .tab { padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #2d3148; background: transparent; color: #94a3b8; cursor: pointer; font-size: 0.875rem; transition: all 0.2s; }
    .tab:hover { border-color: #6366f1; color: #e2e8f0; }
    .tab.active { background: #6366f1; border-color: #6366f1; color: white; font-weight: 600; }
    label { display: block; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.4rem; }
    textarea { width: 100%; background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; color: #e2e8f0; padding: 0.875rem; font-size: 0.9rem; resize: vertical; min-height: 140px; line-height: 1.6; transition: border-color 0.2s; }
    textarea:focus { outline: none; border-color: #6366f1; }
    .lang-row { display: none; margin-top: 0.75rem; }
    .lang-row.visible { display: block; }
    select { background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; color: #e2e8f0; padding: 0.5rem 0.875rem; font-size: 0.875rem; width: 100%; }
    button.run { margin-top: 1rem; width: 100%; padding: 0.75rem; background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none; border-radius: 8px; color: white; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
    button.run:hover { opacity: 0.9; }
    button.run:disabled { opacity: 0.5; cursor: not-allowed; }
    .result-box { display: none; margin-top: 1.5rem; }
    .result-box.visible { display: block; }
    .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .result-header span { font-size: 0.8rem; color: #94a3b8; }
    .copy-btn { background: transparent; border: 1px solid #2d3148; border-radius: 6px; color: #94a3b8; padding: 0.25rem 0.6rem; font-size: 0.75rem; cursor: pointer; transition: all 0.2s; }
    .copy-btn:hover { border-color: #6366f1; color: #e2e8f0; }
    .result-content { background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; padding: 1rem; font-size: 0.9rem; line-height: 1.7; white-space: pre-wrap; color: #e2e8f0; }
    .error { background: #2d1b1b; border: 1px solid #7f1d1d; border-radius: 8px; padding: 0.875rem; color: #fca5a5; font-size: 0.875rem; margin-top: 1rem; display: none; }
    .error.visible { display: block; }
    .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 8px; vertical-align: middle; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .stats { display: flex; gap: 1rem; margin-top: 0.5rem; }
    .stat { background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; padding: 0.5rem 0.875rem; font-size: 0.8rem; color: #94a3b8; }
    .stat strong { color: #6366f1; }
    footer { text-align: center; margin-top: 2rem; color: #475569; font-size: 0.8rem; }
  </style>
</head>
<body>
  <header>
    <h1>AI Text API</h1>
    <p>Résumé · Mots-clés · Traduction · Amélioration — propulsé par Claude</p>
  </header>
  <div class="container">
    <div class="card">
      <div class="tabs">
        <button class="tab active" data-action="summarize">📝 Résumer</button>
        <button class="tab" data-action="extract-keywords">🔑 Mots-clés</button>
        <button class="tab" data-action="translate">🌍 Traduire</button>
        <button class="tab" data-action="improve">✨ Améliorer</button>
      </div>
      <label for="inputText">Texte à traiter</label>
      <textarea id="inputText" placeholder="Colle ton texte ici…"></textarea>
      <div class="lang-row" id="langRow">
        <label for="langSelect">Langue cible</label>
        <select id="langSelect">
          <option value="anglais">Anglais</option>
          <option value="espagnol">Espagnol</option>
          <option value="arabe">Arabe</option>
          <option value="allemand">Allemand</option>
          <option value="italien">Italien</option>
          <option value="portugais">Portugais</option>
          <option value="japonais">Japonais</option>
          <option value="chinois">Chinois (simplifié)</option>
        </select>
      </div>
      <button class="run" id="runBtn">Lancer l'analyse</button>
      <div class="error" id="errorBox"></div>
      <div class="result-box" id="resultBox">
        <div class="result-header">
          <span id="resultLabel">Résultat</span>
          <button class="copy-btn" id="copyBtn">Copier</button>
        </div>
        <div class="result-content" id="resultContent"></div>
        <div class="stats">
          <div class="stat">Entrée : <strong id="statInput">—</strong> car.</div>
          <div class="stat">Modèle : <strong id="statModel">—</strong></div>
        </div>
      </div>
    </div>
    <footer>Projet portfolio — FastAPI + Claude API · Faiza</footer>
  </div>
  <script>
    const actions = {
      "summarize": { label: "Résumé", btn: "Résumer le texte" },
      "extract-keywords": { label: "Mots-clés", btn: "Extraire les mots-clés" },
      "translate": { label: "Traduction", btn: "Traduire" },
      "improve": { label: "Amélioré", btn: "Améliorer le texte" },
    };
    let currentAction = "summarize";
    document.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        currentAction = tab.dataset.action;
        document.getElementById("runBtn").textContent = actions[currentAction].btn;
        document.getElementById("langRow").classList.toggle("visible", currentAction === "translate");
        document.getElementById("resultBox").classList.remove("visible");
        document.getElementById("errorBox").classList.remove("visible");
      });
    });
    document.getElementById("runBtn").addEventListener("click", async () => {
      const text = document.getElementById("inputText").value.trim();
      const btn = document.getElementById("runBtn");
      const errorBox = document.getElementById("errorBox");
      const resultBox = document.getElementById("resultBox");
      errorBox.classList.remove("visible");
      resultBox.classList.remove("visible");
      if (!text) { errorBox.textContent = "Saisis un texte avant de lancer l'analyse."; errorBox.classList.add("visible"); return; }
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>Traitement en cours…';
      try {
        const body = { text };
        if (currentAction === "translate") body.language = document.getElementById("langSelect").value;
        const res = await fetch("/" + currentAction, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Erreur serveur");
        document.getElementById("resultLabel").textContent = actions[currentAction].label;
        document.getElementById("resultContent").textContent = data.result;
        document.getElementById("statInput").textContent = data.input_length;
        document.getElementById("statModel").textContent = data.model;
        resultBox.classList.add("visible");
      } catch (err) {
        errorBox.textContent = "Erreur : " + err.message;
        errorBox.classList.add("visible");
      } finally {
        btn.disabled = false;
        btn.textContent = actions[currentAction].btn;
      }
    });
    document.getElementById("copyBtn").addEventListener("click", () => {
      navigator.clipboard.writeText(document.getElementById("resultContent").textContent).then(() => {
        const btn = document.getElementById("copyBtn");
        btn.textContent = "Copié ✓";
        setTimeout(() => btn.textContent = "Copier", 2000);
      });
    });
  </script>
</body>
</html>"""


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
    return HTMLResponse(content=HTML_PAGE)


@app.get("/health")
async def health():
    return {"status": "ok", "message": "API opérationnelle"}


@app.post("/summarize", response_model=TextResponse)
async def summarize(body: TextInput):
    if len(body.text) < 50:
        raise HTTPException(status_code=400, detail="Texte trop court (minimum 50 caractères)")
    prompt = f"""Résume ce texte en 3 à 5 phrases claires et concises.\nGarde les informations essentielles. Ne rajoute aucun commentaire.\n\nTexte :\n{body.text}\n\nRésumé :"""
    return TextResponse(result=call_claude(prompt), input_length=len(body.text))


@app.post("/extract-keywords", response_model=TextResponse)
async def extract_keywords(body: TextInput):
    if len(body.text) < 30:
        raise HTTPException(status_code=400, detail="Texte trop court (minimum 30 caractères)")
    prompt = f"""Extrais les 5 à 10 mots-clés ou expressions clés de ce texte.\nRetourne-les sous forme de liste simple, un par ligne, sans numérotation ni explication.\n\nTexte :\n{body.text}\n\nMots-clés :"""
    return TextResponse(result=call_claude(prompt), input_length=len(body.text))


@app.post("/translate", response_model=TextResponse)
async def translate(body: TextInput):
    if len(body.text) < 5:
        raise HTTPException(status_code=400, detail="Texte trop court")
    prompt = f"""Traduis ce texte en {body.language}.\nRetourne uniquement la traduction, sans commentaire.\n\nTexte original :\n{body.text}\n\nTraduction en {body.language} :"""
    return TextResponse(result=call_claude(prompt), input_length=len(body.text))


@app.post("/improve", response_model=TextResponse)
async def improve(body: TextInput):
    if len(body.text) < 20:
        raise HTTPException(status_code=400, detail="Texte trop court (minimum 20 caractères)")
    prompt = f"""Améliore ce texte : corrige les fautes, améliore la fluidité et la clarté.\nGarde le sens original. Retourne uniquement le texte amélioré.\n\nTexte :\n{body.text}\n\nTexte amélioré :"""
    return TextResponse(result=call_claude(prompt), input_length=len(body.text))
