"""
Service Q/A : retrieval (index TF-IDF local) + génération via **MLXServe** (API OpenAI-compatible).
Les chiffres viennent du contexte récupéré ; le LLM formule la réponse.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from analytics.src.rag.retrieval import query_index

app = FastAPI(title="SafePerform Analytics QA", version="1.0.0")

MLXSERVE_BASE_URL = os.getenv("MLXSERVE_BASE_URL", "http://127.0.0.1:8088").rstrip("/")
MLXSERVE_MODEL = os.getenv("MLXSERVE_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
MLXSERVE_API_KEY = os.getenv("MLXSERVE_API_KEY", "").strip()
MLXSERVE_TIMEOUT_S = float(
    os.getenv("MLXSERVE_TIMEOUT_S", os.getenv("OLLAMA_TIMEOUT_S", "120"))
)
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "/data/rag")


def _index_files_ready() -> bool:
    idx = Path(RAG_INDEX_DIR)
    return (idx / "vectorizer.joblib").is_file() and (idx / "documents.json").is_file()


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=15)
    team_scope: bool = Field(
        False,
        description="Si true, retrieval sur toute l'index ; sinon filtre sur player_id si fourni.",
    )
    player_id: Optional[int] = Field(None, ge=1)
    player_name: Optional[str] = Field(None, max_length=200)


class QAResponse(BaseModel):
    answer: str
    model: str
    sources: List[Dict[str, Any]]
    llm_ok: bool = Field(
        ...,
        description="True si MLXServe a répondu ; False si timeout / erreur HTTP (réponse de secours).",
    )


def _mlxserve_chat(prompt: str) -> tuple[str, bool]:
    url = f"{MLXSERVE_BASE_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if MLXSERVE_API_KEY:
        headers["Authorization"] = f"Bearer {MLXSERVE_API_KEY}"
    payload: Dict[str, Any] = {
        "model": MLXSERVE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    try:
        with httpx.Client(timeout=MLXSERVE_TIMEOUT_S) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        return (
            f"[MLXServe indisponible ({MLXSERVE_BASE_URL}): {exc}] "
            "Réponds manuellement à partir des sources ci-dessous.",
            False,
        )

    err = data.get("error")
    if err:
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return f"[MLXServe erreur API: {msg}] Réponds manuellement à partir des sources ci-dessous.", False

    choices = data.get("choices") or []
    if not choices:
        return "Réponse vide (pas de choices dans la réponse MLXServe).", False
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    return (content or "Réponse vide du modèle.", True)


@app.get("/health")
def health():
    """Liveness : le process répond (même si l’index RAG n’est pas encore prêt)."""
    idx = Path(RAG_INDEX_DIR)
    has_index = _index_files_ready()
    return {
        "status": "ok",
        "rag_index_dir": str(idx.resolve()),
        "index_ready": has_index,
        "mlxserve_base_url": MLXSERVE_BASE_URL,
        "mlxserve_model": MLXSERVE_MODEL,
        "mlxserve_api_key_set": bool(MLXSERVE_API_KEY),
    }


@app.get("/ready")
def ready():
    """Readiness : 200 seulement si les fichiers d’index RAG sont présents (pour healthcheck Docker)."""
    idx = Path(RAG_INDEX_DIR)
    has_index = _index_files_ready()
    body = {
        "status": "ready" if has_index else "not_ready",
        "index_ready": has_index,
        "rag_index_dir": str(idx.resolve()),
    }
    if not has_index:
        return JSONResponse(status_code=503, content=body)
    return body


@app.post("/qa", response_model=QAResponse)
def qa(body: QARequest):
    idx_path = Path(RAG_INDEX_DIR)
    if not _index_files_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Index RAG absent ou incomplet. Le conteneur analytics-qa relance pipeline + index au "
                "démarrage ; attendre quelques minutes après un premier `docker compose up`, ou vérifier "
                "les logs du service analytics-qa."
            ),
        )

    scope_pid: Optional[int] = None
    if not body.team_scope and body.player_id is not None:
        scope_pid = int(body.player_id)

    try:
        sources = query_index(body.question, str(idx_path), body.top_k, player_id=scope_pid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}") from exc

    if not sources:
        if scope_pid is not None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Aucune entrée d'index pour le joueur id={scope_pid}. "
                    "Vérifier le pipeline + build_rag_index, ou utiliser « toute l'équipe »."
                ),
            )
        raise HTTPException(status_code=503, detail="Aucun document retrouvé.")

    context = "\n\n---\n\n".join(s["text"] for s in sources)
    scope_lines = ""
    if scope_pid is not None:
        label = (body.player_name or "").strip() or f"joueur id={scope_pid}"
        scope_lines = f"""- La question cible UNIQUEMENT le joueur : {label} (player_id={scope_pid}).
- Réponds pour ce joueur seul ; n'invente pas d'autres joueurs.
- Utilise son nom tel qu'il apparaît dans le contexte (prénom + nom), pas seulement l'id numérique.
"""
    else:
        scope_lines = """- Tu peux comparer plusieurs joueurs si le contexte en contient plusieurs.
- Utilise les noms affichés dans le contexte (ligne « Joueur: … ») autant que possible, avec player_id entre parenthèses si utile.
"""

    prompt = f"""Tu es un analyste performance rugby/staff. Tu réponds en français.

Règles strictes:
- Base-toi UNIQUEMENT sur le contexte ci-dessous pour les faits (chiffres, joueurs, métriques).
- Si le contexte ne permet pas de répondre, dis-le clairement.
{scope_lines}- Ne invente pas de données absentes du contexte.
- Réponse concise (liste à puces autorisée).

Contexte:
{context}

Question: {body.question}
"""
    answer, llm_ok = _mlxserve_chat(prompt)
    return QAResponse(
        answer=answer,
        model=MLXSERVE_MODEL,
        sources=sources,
        llm_ok=llm_ok,
    )
