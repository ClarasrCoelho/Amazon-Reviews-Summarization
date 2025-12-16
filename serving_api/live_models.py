# serving_api/live_models.py
from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List

from serving_api.s3_utils import read_text_from_s3, write_text_to_s3
from libs.nlp_udfs import predict_sentiment_batch, extract_aspects_batch
from libs.summarize import summarize_hybrid

S3_BUCKET = os.getenv("S3_BUCKET", "labgdd-tp2-2")
S3_PREFIX = os.getenv("S3_PREFIX", "labgdd")

USER_REVIEWS_PREFIX = f"{S3_PREFIX}/gold/user_reviews/"
LIVE_BOOKS_PREFIX = f"{S3_PREFIX}/gold/live/books/"


def _user_reviews_key(asin: str) -> str:
    return f"{USER_REVIEWS_PREFIX}{asin}.jsonl"


def _live_book_key(asin: str) -> str:
    return f"{LIVE_BOOKS_PREFIX}{asin}.json"


def load_user_reviews(asin: str) -> List[dict]:
    """
    Lê as reviews 'live' do S3 (jsonl) para um ASIN.
    """
    try:
        txt = read_text_from_s3(S3_BUCKET, _user_reviews_key(asin))
    except Exception:
        return []

    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _concat_texts(reviews: List[dict], max_chars: int = 200_000) -> str:
    parts = []
    for r in reviews:
        t = (r.get("text") or "").strip()
        if t:
            parts.append(t)
    s = " ".join(parts)
    return s[:max_chars]


def run_live_models(asin: str) -> Dict[str, Any]:
    """
    Corre 'modelos' em tempo real com base nas user_reviews:
    - sentimento médio (usando o teu predict_sentiment_batch)
    - aspetos agregados (extract_aspects_batch)
    - resumo (summarize_hybrid)
    Guarda resultado em: gold/live/books/{asin}.json
    """
    reviews = load_user_reviews(asin)
    text_all = _concat_texts(reviews)

    # Sentiment: usa o batch util (pd.Series), mas aqui chamamos de forma simples
    # Fazemos um "batch" de 1 (texto concatenado)
    import pandas as pd
    sent = float(predict_sentiment_batch(pd.Series([text_all]))[0]) if text_all else 0.0

    # Aspects: também batch de 1
    aspects = extract_aspects_batch(pd.Series([text_all]))[0] if text_all else []
    if aspects is None:
        aspects = []
    if not isinstance(aspects, list):
        try:
            aspects = list(aspects)
        except Exception:
            aspects = []

    # Summary (híbrido)
    summary = summarize_hybrid(text_all, num_sentences_textrank=5, max_new_tokens=128) if text_all else ""

    result = {
        "asin": asin,
        "n_user_reviews": len(reviews),
        "summary_live": summary,
        "avg_sentiment_live": sent,
        "aspects_live": aspects[:20],
        "ts": time.time(),
    }

    write_text_to_s3(
        S3_BUCKET,
        _live_book_key(asin),
        json.dumps(result, ensure_ascii=False),
        content_type="application/json",
    )
    return result
