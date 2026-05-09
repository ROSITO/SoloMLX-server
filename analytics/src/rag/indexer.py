from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def build_and_save_index(documents: List[Dict], output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts = [d["text"] for d in documents]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)

    nn = NearestNeighbors(metric="cosine")
    nn.fit(matrix)

    joblib.dump(vectorizer, out / "vectorizer.joblib")
    joblib.dump(nn, out / "nn.joblib")
    joblib.dump(matrix, out / "matrix.joblib")
    (out / "documents.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"documents": len(documents), "vocab_size": len(vectorizer.vocabulary_)}
