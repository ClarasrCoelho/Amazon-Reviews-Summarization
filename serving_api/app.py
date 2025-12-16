import os
import json
import time
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from serving_api.schemas import ReviewIn
from serving_api.s3_utils import read_text_from_s3, list_keys, append_jsonl, write_text_to_s3

from libs.nlp_udfs import predict_sentiment_batch, extract_aspects_batch, simple_aspect_polarity
from libs.summarize import summarize_hybrid

app = FastAPI(title="LABGDD – Amazon Books Replica", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

S3_BUCKET = os.getenv("S3_BUCKET", "labgdd-tp2-2")
S3_PREFIX = os.getenv("S3_PREFIX", "labgdd")

CATALOG_PREFIX = f"{S3_PREFIX}/gold/catalog/books_index/"
USER_REVIEWS_PREFIX = f"{S3_PREFIX}/gold/user_reviews/"
LIVE_BOOKS_PREFIX = f"{S3_PREFIX}/gold/live/books/"
LIVE_RUNS_PREFIX = f"{S3_PREFIX}/gold/live/model_runs/"

_catalog_cache = {"ts": 0.0, "items": []}
CATALOG_TTL_SECONDS = 30.0

MAX_LIVE_CHARS = 60_000  # para não rebentar com concat enorme


def _norm(s: Any) -> str:
    return str(s).strip()


def _load_catalog_from_s3() -> List[dict]:
    keys = [k for k in list_keys(S3_BUCKET, CATALOG_PREFIX) if k.endswith(".json")]
    items: List[dict] = []
    for k in keys:
        txt = read_text_from_s3(S3_BUCKET, k)
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def get_catalog() -> List[dict]:
    now = time.time()
    if (now - _catalog_cache["ts"]) < CATALOG_TTL_SECONDS and _catalog_cache["items"]:
        return _catalog_cache["items"]
    items = _load_catalog_from_s3()
    _catalog_cache["ts"] = now
    _catalog_cache["items"] = items
    return items


def user_reviews_key(asin: str) -> str:
    return f"{USER_REVIEWS_PREFIX}{asin}.jsonl"


def live_book_key(asin: str) -> str:
    return f"{LIVE_BOOKS_PREFIX}{asin}.json"


def live_runs_key(asin: str) -> str:
    return f"{LIVE_RUNS_PREFIX}{asin}.jsonl"


def load_user_reviews(asin: str) -> List[dict]:
    key = user_reviews_key(asin)
    try:
        txt = read_text_from_s3(S3_BUCKET, key)
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


def load_live_book(asin: str) -> Dict[str, Any] | None:
    key = live_book_key(asin)
    try:
        txt = read_text_from_s3(S3_BUCKET, key)
        return json.loads(txt)
    except Exception:
        return None


def _concat_user_reviews_text(revs: List[dict]) -> str:
    # últimas 200 reviews para o live (ajusta se quiseres)
    texts = []
    for r in revs[-200:]:
        t = (r.get("text") or "").strip()
        if t:
            texts.append(t)
    joined = " ".join(texts).strip()
    if len(joined) > MAX_LIVE_CHARS:
        joined = joined[:MAX_LIVE_CHARS]
    return joined


def run_live_models_for_asin(asin: str, user_reviews: List[dict]) -> Dict[str, Any]:
    t0 = time.time()

    text = _concat_user_reviews_text(user_reviews)

    # 1) Sentiment (score [-1,1])
    sent = float(predict_sentiment_batch(pd.Series([text])).iloc[0]) if text else 0.0

    # 2) Aspects + polarity (simples, offline-friendly)
    aspects = extract_aspects_batch(pd.Series([text])).iloc[0] if text else []
    aspects = aspects if isinstance(aspects, list) else []
    # polaridade média pelos aspects encontrados
    if aspects:
        pols = [simple_aspect_polarity(text, a) for a in aspects[:8]]
        avg_aspect_polarity = float(sum(pols) / max(len(pols), 1))
    else:
        avg_aspect_polarity = 0.0

    # 3) Summary (híbrido: TextRank + (T5 se disponível))
    summary = summarize_hybrid(text, num_sentences_textrank=5, max_new_tokens=128) if text else ""

    dt = time.time() - t0
    return {
        "asin": asin,
        "updated_at": time.time(),
        "inputs": {
            "user_reviews_used": min(len(user_reviews), 200),
            "chars_used": len(text),
        },
        "models": {
            "sentiment_live": {
                "value": sent,
                "range": [-1, 1],
            },
            "absa_live": {
                "all_aspects": aspects[:50],
                "avg_aspect_polarity": avg_aspect_polarity,
                "range": [-1, 1],
            },
            "summary_live": {
                "text": summary,
                "strategy": "TextRank + (T5 if available)",
            },
        },
        "timing_sec": dt,
    }


def append_model_run_log(asin: str, payload: Dict[str, Any]):
    rec = {
        "asin": asin,
        "ts": time.time(),
        "payload": payload,
    }
    append_jsonl(S3_BUCKET, live_runs_key(asin), rec)


@app.get("/books")
def list_books(
    query: str | None = Query(default=None, description="Pesquisa por título/asin"),
    limit: int = Query(default=30, ge=1, le=100),
):
    books = get_catalog()
    if query:
        q = query.strip().lower()
        books = [
            b for b in books
            if q in str(b.get("title", "")).lower() or q in str(b.get("asin", "")).lower()
        ]

    def score(b):
        return (b.get("review_count") or 0, b.get("avg_rating") or 0.0)

    books = sorted(books, key=score, reverse=True)
    return books[:limit]


@app.get("/books/{asin}")
def get_book_page(asin: str):
    asin = _norm(asin)
    books = get_catalog()

    book = next((b for b in books if _norm(b.get("asin", "")) == asin), None)
    if not book:
        return {"error": "Livro não encontrado", "asin": asin}

    user_reviews = load_user_reviews(asin)
    live = load_live_book(asin)

    # summary mostrado no topo:
    # 1) summary_live se existir e tiver texto
    # 2) senão, book_summary do catálogo (gold/serving)
    summary_for_reviews = ""
    if live and live.get("models", {}).get("summary_live", {}).get("text"):
        summary_for_reviews = live["models"]["summary_live"]["text"]
    else:
        summary_for_reviews = book.get("book_summary") or ""

    return {
        "book": book,
        "summary_for_reviews": summary_for_reviews,
        "user_reviews": list(reversed(user_reviews))[:50],
        "live_available": bool(live),
    }


@app.get("/models/{asin}")
def get_models_live(asin: str):
    asin = _norm(asin)
    live = load_live_book(asin)
    if not live:
        return {
            "asin": asin,
            "status": "no_live_results",
            "message": "Ainda não há resultados live. Adiciona uma review.",
        }
    return live


@app.post("/reviews")
def add_review(review: ReviewIn):
    asin = _norm(review.asin)

    record = {
        "asin": asin,
        "text": review.text,
        "rating": review.rating,
        "user": review.user,
        "ts": time.time(),
    }

    # 1) guardar review no S3 (append)
    append_jsonl(S3_BUCKET, user_reviews_key(asin), record)

    # 2) recarregar user reviews e correr “modelos live”
    user_reviews = load_user_reviews(asin)
    live_payload = run_live_models_for_asin(asin, user_reviews)

    # 3) escrever snapshot live + log de runs
    write_text_to_s3(
        S3_BUCKET,
        live_book_key(asin),
        json.dumps(live_payload, ensure_ascii=False),
        content_type="application/json",
    )
    append_model_run_log(asin, live_payload)

    return {"status": "ok", "saved": record, "live": live_payload}


@app.get("/models")
def models_info():
    return {
        "pipelines": [
            "catalog (gold/catalog/books_index)",
            "user reviews (gold/user_reviews/{asin}.jsonl)",
            "live snapshot (gold/live/books/{asin}.json)",
            "live runs log (gold/live/model_runs/{asin}.jsonl)",
        ],
        "note": "Usa /models/{asin} para veres resultados live por livro.",
    }
