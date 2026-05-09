from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from sklearn.metrics.pairwise import cosine_similarity


def query_index(
    question: str,
    index_dir: str,
    top_k: int = 5,
    player_id: Optional[int] = None,
) -> List[Dict]:
    idx = Path(index_dir)
    vectorizer = joblib.load(idx / "vectorizer.joblib")
    nn = joblib.load(idx / "nn.joblib")
    matrix = joblib.load(idx / "matrix.joblib")
    documents = json.loads((idx / "documents.json").read_text(encoding="utf-8"))

    qv = vectorizer.transform([question])

    if player_id is not None:
        pid = int(player_id)
        matching = [i for i, doc in enumerate(documents) if int(doc.get("player_id") or -1) == pid]
        if not matching:
            return []
        scored: List[tuple[float, int]] = []
        for i in matching:
            sim = float(cosine_similarity(qv, matrix[int(i)])[0, 0])
            scored.append((sim, int(i)))
        scored.sort(key=lambda x: -x[0])
        out: List[Dict] = []
        for sim, i in scored[: max(1, top_k)]:
            doc = documents[i]
            out.append(_doc_hit(doc, sim))
        return out

    distances, indices = nn.kneighbors(qv, n_neighbors=min(top_k, matrix.shape[0]))
    out = []
    for dist, i in zip(distances[0], indices[0]):
        doc = documents[int(i)]
        out.append(_doc_hit(doc, float(1 - dist)))
    return out


def _doc_hit(doc: Dict, score: float) -> Dict:
    return {
        "doc_id": doc["doc_id"],
        "player_id": doc.get("player_id"),
        "player_display": doc.get("player_display"),
        "score": score,
        "text": doc["text"],
    }
