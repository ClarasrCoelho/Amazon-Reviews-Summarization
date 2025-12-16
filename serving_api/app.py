import os
import json
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from serving_api.schemas import ReviewIn
from serving_api.s3_utils import read_text_from_s3, list_keys, append_jsonl

app = FastAPI(title="LABGDD – Amazon Books Replica", version="0.2.0")

# CORS para frontend estático (file:// ou http server local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config S3 / paths ----
S3_BUCKET = os.getenv("S3_BUCKET", "labgdd-tp2-2")
S3_PREFIX = os.getenv("S3_PREFIX", "labgdd")

# Catálogo leve (criado pelo job)
CATALOG_PREFIX = f"{S3_PREFIX}/gold/catalog/books_index/"
# Reviews "live"
USER_REVIEWS_PREFIX = f"{S3_PREFIX}/gold/user_reviews/"

# Em debug: se quiseres não bater no S3
DISABLE_S3 = os.getenv("DISABLE_S3", "0") == "1"

# cache simples em memória (para não ires ao S3 em cada request)
_catalog_cache = {"ts": 0.0, "items": []}
CATALOG_TTL_SECONDS = 30.0


def _load_catalog_from_s3() -> list[dict]:
    # lê todos os part-*.json do prefix e concatena
    keys = [k for k in list_keys(S3_BUCKET, CATALOG_PREFIX) if k.endswith(".json")]
    items = []
    for k in keys:
        txt = read_text_from_s3(S3_BUCKET, k)
        # spark json writer: 1 json por linha
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def get_catalog() -> list[dict]:
    if DISABLE_S3:
        # catálogo mínimo para testar sem S3
        return [
            {
                "asin": "TEST123",
                "title": "Livro TEST123",
                "imUrl": "",
                "price": None,
                "categories": [],
                "book_summary": "Resumo de teste (DISABLE_S3=1).",
                "avg_sentiment": 0.1,
                "avg_rating": 4.2,
                "review_count": 12,
                "all_aspects": ["writing", "story"],
                "avg_aspect_polarity": 0.05,
            }
        ]

    now = time.time()
    if (now - _catalog_cache["ts"]) < CATALOG_TTL_SECONDS and _catalog_cache["items"]:
        return _catalog_cache["items"]

    items = _load_catalog_from_s3()
    _catalog_cache["ts"] = now
    _catalog_cache["items"] = items
    return items


def user_reviews_key(asin: str) -> str:
    return f"{USER_REVIEWS_PREFIX}{asin}.jsonl"


def load_user_reviews(asin: str) -> list[dict]:
    if DISABLE_S3:
        return []

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

    # ordena: mais reviews primeiro (ou podes ordenar por avg_rating)
    def score(b):
        return (b.get("review_count") or 0, b.get("avg_rating") or 0.0)

    books = sorted(books, key=score, reverse=True)
    return books[:limit]


@app.get("/books/{asin}")
def get_book_page(asin: str):
    books = get_catalog()
    book = next((b for b in books if b.get("asin") == asin), None)
    if not book:
        return {"error": "Livro não encontrado", "asin": asin}

    # reviews "live"
    user_reviews = load_user_reviews(asin)

    # Nota: aqui estás a mostrar “summary do livro” (gold/serving) no topo das reviews
    # As reviews do dataset original podes integrar mais tarde com outro índice,
    # mas para demo: já dá para ver as user_reviews em tempo real.
    return {
        "book": book,
        "summary_for_reviews": book.get("book_summary") or "",
        "user_reviews": list(reversed(user_reviews))[:50],
    }


@app.post("/reviews")
def add_review(review: ReviewIn):
    record = {
        "asin": review.asin,
        "text": review.text,
        "rating": review.rating,
        "user": review.user,
        "ts": time.time(),
    }

    if DISABLE_S3:
        return {"status": "ok (DISABLE_S3=1)", "saved": record}

    append_jsonl(S3_BUCKET, user_reviews_key(review.asin), record)
    return {"status": "ok", "saved": record}


@app.get("/models")
def models_info():
    return {
        "pipelines": [
            "sentiment (gold/sentiment)",
            "absa (gold/absa)",
            "summaries (gold/summaries)",
            "serving join (gold/serving)",
            "books_index (gold/catalog/books_index)",
        ],
        "note": "Mostra aqui métricas/prints/explicações dos modelos (podes ligar a outputs do teu projeto).",
    }
