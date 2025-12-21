import os
import json
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Query, HTTPException, Request, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from serving_api.schemas import ReviewCreate
from serving_api.s3_utils import read_text_from_s3, list_keys, append_jsonl

import re
from collections import Counter

import subprocess
import sys


# Forçar UTF-8 em tudo o que for jobs em background (especialmente Windows)
_BASE_ENV = os.environ.copy()
_BASE_ENV["PYTHONIOENCODING"] = "utf-8"
_BASE_ENV["PYTHONUTF8"] = "1"


# -----------------------------
# Tokenizers / metrics helpers
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")
_ROUGE_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def summary_quality_metrics(summary: str, source_text: str) -> dict:
    sum_toks = _words(summary)
    src_toks = _words(source_text)

    sum_words = len(sum_toks)
    src_words = len(src_toks)

    compression = (sum_words / src_words) if src_words else None

    bigrams = _ngrams(sum_toks, 2)
    red = 0.0
    if bigrams:
        c = Counter(bigrams)
        red = 1.0 - (len(c) / len(bigrams))

    novelty = None
    if sum_toks:
        src_set = set(src_toks)
        novelty = sum(1 for t in sum_toks if t not in src_set) / len(sum_toks)

    return {
        "summary_words": sum_words,
        "source_words": src_words,
        "compression_ratio": compression,
        "redundancy_score": red,
        "novelty_score": novelty,
    }


def _tok(text: str) -> list[str]:
    return _ROUGE_WORD_RE.findall((text or "").lower())


def rouge_n_f1(hyp: str, ref: str, n: int = 1) -> float | None:
    ht = _tok(hyp)
    rt = _tok(ref)
    if len(ht) < n or len(rt) < n:
        return None

    def ngrams(toks):
        return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]

    from collections import Counter as _C

    h = _C(ngrams(ht))
    r = _C(ngrams(rt))

    overlap = sum((h & r).values())
    if overlap == 0:
        return 0.0

    denom_h = sum(h.values())
    denom_r = sum(r.values())
    if denom_h == 0 or denom_r == 0:
        return None

    prec = overlap / denom_h
    rec = overlap / denom_r
    if (prec + rec) == 0:
        return None
    return 2 * prec * rec / (prec + rec)


def rouge_l_f1(hyp: str, ref: str) -> float | None:
    a = _tok(hyp)
    b = _tok(ref)
    if not a or not b:
        return None

    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp

    lcs = dp[m]
    prec = lcs / n if n else None
    rec = lcs / m if m else None
    if prec is None or rec is None or (prec + rec) == 0:
        return None
    return 2 * prec * rec / (prec + rec)


# -----------------------------
# Windows-safe background launcher (NÃO abre CMD)
# -----------------------------
def _popen_detached_no_window(cmd: list[str], log, env: dict) -> subprocess.Popen:
    """
    Em Windows, isto evita abrir uma janela (CMD) mesmo quando o uvicorn foi arrancado sem consola/IDE.
    """
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0

        return subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            cwd=os.getcwd(),
            env=env,
            shell=False,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )

    return subprocess.Popen(
        cmd,
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        cwd=os.getcwd(),
        env=env,
        close_fds=True,
    )


def run_process_new_reviews_job(asin: str):
    try:
        os.makedirs("logs/serving_jobs", exist_ok=True)
        log_path = os.path.join("logs", "serving_jobs", f"{asin}.log")

        env = _BASE_ENV.copy()
        py = sys.executable  # usa sempre a venv

        cmd = [
            py, "-m", "jobs.process_new_reviews",
            "--asin", asin,
            "--min_new_reviews", "1",
            "--hours_back", "24",
        ]

        with open(log_path, "a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n--- BG RUN {datetime.now().isoformat()} asin={asin} ---\n")
            log.write(f"cwd={os.getcwd()}\n")
            log.write(f"py={py}\n")
            log.write("cmd=" + " ".join(cmd) + "\n")

            p = _popen_detached_no_window(cmd, log=log, env=env)
            log.write(f"pid={p.pid}\n")

    except Exception:
        pass


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="LABGDD – Amazon Books Replica", version="0.3.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{path:path}")
def preflight(path: str, request: Request):
    origin = request.headers.get("origin")
    allowed = {
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    }
    headers = {
        "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
        "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers", "*"),
        "Access-Control-Max-Age": "86400",
    }
    if origin in allowed:
        headers["Access-Control-Allow-Origin"] = origin
    return Response(status_code=204, headers=headers)


# -----------------------------
# S3 settings
# -----------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "labgdd-tp2-2")
S3_PREFIX = os.getenv("S3_PREFIX", "labgdd").strip("/")

CATALOG_PREFIX = f"{S3_PREFIX}/gold/catalog/books_index/"
USER_REVIEWS_PREFIX = f"{S3_PREFIX}/gold/user_reviews/"
SERVING_BOOK_PREFIX = f"{S3_PREFIX}/gold/serving/books/"
BRONZE_REVIEWS_PREFIX = f"{S3_PREFIX}/bronze/reviews_events/"

EVAL_PREFIX_MAIN = f"{S3_PREFIX}/gold/eval/"
EVAL_PREFIX_LEGACY = f"{S3_PREFIX}/gold/eval/model_metrics/"

EVAL_LOCK_TS = 0.0
EVAL_LOCK_SECONDS = 300.0  # 5 min


# ------------- helpers -------------
def serving_book_key(asin: str) -> str:
    return f"{SERVING_BOOK_PREFIX}{asin}.json"


def try_load_serving_book(asin: str) -> dict | None:
    try:
        txt = read_text_from_s3(S3_BUCKET, serving_book_key(asin))
        txt = (txt or "").strip()
        if not txt:
            return {"_invalid": True, "_error": "empty_file"}
        try:
            return json.loads(txt)
        except Exception as e:
            return {
                "_invalid": True,
                "_error": f"invalid_json:{type(e).__name__}",
                "_head": txt[:500],
            }
    except Exception:
        return None


def user_reviews_key(asin: str) -> str:
    return f"{USER_REVIEWS_PREFIX}{asin}.jsonl"


def bronze_reviews_key(dt: str, hour: str) -> str:
    return f"{BRONZE_REVIEWS_PREFIX}dt={dt}/hour={hour}/reviews.jsonl"


# -----------------------------
# Catálogo (cache)
# -----------------------------
_catalog_cache = {"ts": 0.0, "items": [], "by_asin": {}}
CATALOG_TTL_SECONDS = 300.0  # 5 min


def _load_catalog_from_s3() -> list[dict]:
    prefixes = [CATALOG_PREFIX.rstrip("/"), CATALOG_PREFIX.rstrip("/") + "/"]
    all_keys: list[str] = []
    last_err: str | None = None

    for pref in prefixes:
        try:
            ks = list_keys(S3_BUCKET, pref) or []
            all_keys.extend(ks)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    all_keys = list(dict.fromkeys(all_keys))
    keys = [
        k
        for k in all_keys
        if (k.endswith(".json") or k.endswith(".jsonl"))
        and ("/_" not in k)
        and (not k.split("/")[-1].startswith("_"))
    ]

    if not keys:
        raise HTTPException(
            status_code=500,
            detail={
                "msg": "Não encontrei ficheiros do catálogo no S3.",
                "bucket": S3_BUCKET,
                "tried_prefixes": prefixes,
                "keys_seen_count": len(all_keys),
                "example_keys": all_keys[:10],
                "last_err": last_err,
            },
        )

    items: list[dict] = []
    for k in keys:
        try:
            txt = read_text_from_s3(S3_BUCKET, k)
        except Exception:
            continue

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
    now = time.time()
    if (now - _catalog_cache["ts"]) < CATALOG_TTL_SECONDS and _catalog_cache["items"]:
        return _catalog_cache["items"]

    items = _load_catalog_from_s3()
    _catalog_cache["ts"] = now
    _catalog_cache["items"] = items
    _catalog_cache["by_asin"] = {
        str(b.get("asin", "")).strip(): b for b in items if b.get("asin") is not None
    }
    return items


def get_book_from_catalog(asin: str) -> dict | None:
    get_catalog()
    return _catalog_cache["by_asin"].get(asin.strip())


def load_user_reviews(asin: str) -> list[dict]:
    key = user_reviews_key(asin)
    try:
        txt = read_text_from_s3(S3_BUCKET, key)
    except Exception:
        return []
    out: list[dict] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def append_user_review(asin: str, review: dict):
    append_jsonl(S3_BUCKET, user_reviews_key(asin), review)


# -----------------------------
# MÉTRICAS: rating (batch + live)
# -----------------------------
def _rating_metrics(book: dict | None, user_reviews: list[dict]) -> dict:
    base_avg = safe_float((book or {}).get("avg_rating"))
    try:
        base_count = int((book or {}).get("review_count") or 0)
    except Exception:
        base_count = 0

    live_ratings = []
    for r in user_reviews:
        rr = safe_float(r.get("rating"))
        if rr is not None and 1.0 <= rr <= 5.0:
            live_ratings.append(rr)

    live_count = len(live_ratings)
    live_sum = float(sum(live_ratings)) if live_ratings else 0.0
    global_count = base_count + live_count

    if global_count > 0:
        if base_avg is not None and base_count > 0:
            global_avg = (base_avg * base_count + live_sum) / global_count
        elif live_count > 0:
            global_avg = live_sum / live_count
        else:
            global_avg = base_avg
    else:
        global_avg = base_avg

    delta_rating = None
    if base_avg is not None and global_avg is not None:
        delta_rating = float(global_avg) - float(base_avg)

    return {
        "base_avg": base_avg,
        "base_count": base_count,
        "live_count_valid": live_count,
        "global_avg": float(global_avg) if global_avg is not None else None,
        "global_count": global_count,
        "delta_rating": delta_rating,
    }


# -----------------------------
# ✅ NOVO: MÉTRICAS sentimento (batch + live)
# -----------------------------
def _sentiment_proxy_from_rating(rating: float | None) -> float | None:
    """
    Proxy simples e consistente:
      rating 1..5 -> sentimento em [-1, 1]
      1 -> -1, 3 -> 0, 5 -> +1
    """
    rr = safe_float(rating)
    if rr is None:
        return None
    if rr < 1.0 or rr > 5.0:
        return None
    return float((rr - 3.0) / 2.0)


def _sentiment_metrics(book: dict | None, user_reviews: list[dict]) -> dict:
    base_sent = safe_float((book or {}).get("avg_sentiment"))
    try:
        base_count = int((book or {}).get("review_count") or 0)
    except Exception:
        base_count = 0

    live_sents: list[float] = []
    for r in user_reviews:
        ss = safe_float(r.get("sentiment_score"))
        if ss is None:
            # compat: se houver reviews antigas sem sentiment_score, tenta derivar do rating
            ss = _sentiment_proxy_from_rating(r.get("rating"))
        if ss is None:
            continue
        # normaliza/valida para [-1,1]
        if ss < -1.0:
            ss = -1.0
        if ss > 1.0:
            ss = 1.0
        live_sents.append(float(ss))

    live_count = len(live_sents)
    live_sum = float(sum(live_sents)) if live_sents else 0.0
    global_count = base_count + live_count

    if global_count > 0:
        if base_sent is not None and base_count > 0:
            global_sent = (base_sent * base_count + live_sum) / global_count
        elif live_count > 0:
            global_sent = live_sum / live_count
        else:
            global_sent = base_sent
    else:
        global_sent = base_sent

    delta_sent = None
    if base_sent is not None and global_sent is not None:
        delta_sent = float(global_sent) - float(base_sent)

    return {
        "base_sentiment": base_sent,
        "base_count": base_count,
        "live_count_valid": live_count,
        "global_sentiment": float(global_sent) if global_sent is not None else None,
        "global_count": global_count,
        "delta_sentiment": delta_sent,
    }


def _get_book_page_payload(asin: str) -> dict:
    asin = asin.strip()
    book = get_book_from_catalog(asin)

    batch_summary = None
    if book:
        batch_summary = book.get("book_summary") or book.get("summary")

    user_reviews = load_user_reviews(asin)

    rating_stats = _rating_metrics(book, user_reviews)
    sentiment_stats = _sentiment_metrics(book, user_reviews)

    serving = try_load_serving_book(asin)

    final_summary = None
    if serving and not serving.get("_invalid"):
        final_summary = serving.get("summary") or serving.get("final_summary")

    return {
        "asin": asin,
        "book": book,
        "summary": final_summary or batch_summary,
        "user_reviews": user_reviews,
        "rating_stats": rating_stats,
        "sentiment_stats": sentiment_stats,
        "serving": serving,
    }


# -----------------------------
# Throttle serving job
# -----------------------------
SERVING_LOCK: dict[str, float] = {}
SERVING_LOCK_SECONDS = 120.0


def can_fire_serving(asin: str) -> bool:
    now = time.time()
    last = SERVING_LOCK.get(asin, 0.0)
    if (now - last) < SERVING_LOCK_SECONDS:
        return False
    SERVING_LOCK[asin] = now
    return True


# -----------------------------
# Routes
# -----------------------------
@app.get("/books")
def list_books(
    q: str = Query(default="", description="Pesquisa por título/asin"),
    limit: int = Query(default=60, ge=1, le=500),
):
    books = get_catalog()

    if q:
        ql = q.lower().strip()
        books = [
            b for b in books
            if ql in str(b.get("title", "")).lower()
            or ql in str(b.get("asin", "")).lower()
        ]

    def _pop(b):
        try:
            return float(b.get("review_count") or 0)
        except Exception:
            return 0.0

    books = sorted(books, key=_pop, reverse=True)
    return books[:limit]


@app.get("/books/{asin}")
def get_book_page(asin: str):
    return _get_book_page_payload(asin)


@app.post("/books/{asin}/reviews")
def add_review(asin: str, review: ReviewCreate, background_tasks: BackgroundTasks):
    asin = asin.strip()
    now = datetime.now(timezone.utc)

    # ✅ sentimento proxy calculado já aqui
    sent_proxy = _sentiment_proxy_from_rating(float(review.rating))

    rec = {
        "ts": time.time(),
        "dt": now.strftime("%Y-%m-%d"),
        "hour": now.strftime("%H"),
        "asin": asin,
        "user": review.user,
        "rating": float(review.rating),
        "sentiment_score": sent_proxy,   # ✅ novo
        "text": review.text,
        "source": "web",
    }

    # 1) bronze event
    bronze_key = bronze_reviews_key(dt=rec["dt"], hour=rec["hour"])
    append_jsonl(S3_BUCKET, bronze_key, rec)

    # 2) user_reviews (para aparecer imediatamente no site)
    append_user_review(
        asin,
        {
            "ts": rec["ts"],
            "user": rec["user"],
            "rating": rec["rating"],
            "sentiment_score": rec["sentiment_score"],  # ✅ novo
            "text": rec["text"],
        },
    )

    # 3) lançar job (sem crashar o endpoint)
    job_fired = False
    throttled = False
    log_path = None

    if not can_fire_serving(asin):
        throttled = True
    else:
        job_fired = True
        # background_tasks.add_task(run_process_new_reviews_job, asin)
        log_path = os.path.join("logs", "serving_jobs", f"{asin}.log")

    return {
        "ok": True,
        "asin": asin,
        "review": rec,
        "job_fired": job_fired,
        "log_path": log_path,
        "job_throttled": throttled,
    }


@app.get("/serving_status/{asin}")
def serving_status(asin: str):
    asin = asin.strip()
    key = serving_book_key(asin)
    try:
        txt = read_text_from_s3(S3_BUCKET, key)
        ready = bool((txt or "").strip())
        return {"asin": asin, "exists": True, "ready": ready}
    except Exception as e:
        return {"asin": asin, "exists": False, "ready": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/debug/serving/{asin}")
def debug_serving(asin: str):
    asin = asin.strip()
    key = serving_book_key(asin)
    try:
        txt = read_text_from_s3(S3_BUCKET, key)
        head = (txt or "")[:800]
        try:
            info = json.loads(txt) if (txt or "").strip() else None
            return {
                "exists": True,
                "bucket": S3_BUCKET,
                "key": key,
                "bytes": len(txt or ""),
                "json_ok": True,
                "parsed_keys": list(info.keys())[:30] if isinstance(info, dict) else None,
            }
        except Exception as e:
            return {
                "exists": True,
                "bucket": S3_BUCKET,
                "key": key,
                "bytes": len(txt or ""),
                "json_ok": False,
                "json_error": f"{type(e).__name__}: {e}",
                "head": head,
            }
    except Exception as e:
        return {"exists": False, "bucket": S3_BUCKET, "key": key, "error": f"{type(e).__name__}: {e}"}


@app.get("/models/{asin}")
def models_for_asin(asin: str):
    asin = asin.strip()

    serving = try_load_serving_book(asin)
    serving_present = serving is not None
    serving_invalid = bool(serving_present and isinstance(serving, dict) and serving.get("_invalid"))
    serving_available = serving_present and not serving_invalid

    book = get_book_from_catalog(asin)
    batch_summary = (book.get("book_summary") or book.get("summary")) if book else None

    user_reviews = load_user_reviews(asin)
    live_texts = [(r.get("text") or "").strip() for r in user_reviews if (r.get("text") or "").strip()]

    rating_stats = _rating_metrics(book, user_reviews)
    sentiment_stats = _sentiment_metrics(book, user_reviews)

    serving_summary = None
    serving_ts = None
    source_for_summary = " ".join(live_texts)

    if serving_available:
        serving_summary = serving.get("summary") or serving.get("final_summary")
        serving_ts = serving.get("updated_ts") or serving.get("ts")
        source_for_summary = serving.get("source_text") or source_for_summary

    freshness = None
    if serving_ts is not None:
        try:
            freshness = time.time() - float(serving_ts)
        except Exception:
            freshness = None

    if not serving_summary:
        summary_quality = {"status": "missing_serving", "note": "Ainda não existe serving_summary."}
        rouge_proxy = None
    else:
        summary_quality = summary_quality_metrics(serving_summary, source_for_summary or "")
        summary_quality["freshness_seconds"] = freshness
        rouge_proxy = None
        if batch_summary:
            rouge_proxy = {
                "rouge1_f1": rouge_n_f1(serving_summary, batch_summary, 1),
                "rouge2_f1": rouge_n_f1(serving_summary, batch_summary, 2),
                "rougeL_f1": rouge_l_f1(serving_summary, batch_summary),
            }

    out = {
        "asin": asin,
        "serving_available": serving_available,
        "serving_present": serving_present,
        "serving_invalid": serving_invalid,
        "summaries": {
            "batch_summary": batch_summary,
            "serving_summary": serving_summary,
        },
        "metrics": {
            "rating": rating_stats,
            # ✅ agora sentimento tem base + live + global
            "sentiment": sentiment_stats,
            "summary_quality": summary_quality,
            "summary_eval": {
                "rouge_proxy": rouge_proxy,
                "note": "Proxy: ROUGE(serving_summary vs batch_summary). Só existe quando ambos existem.",
            },
            "streaming": {
                "user_reviews_total": len(user_reviews),
                "user_reviews_with_text": len(live_texts),
            },
        },
        "serving": serving,
    }

    if not serving_present:
        out["warning"] = "Ainda não há output de SERVING para este ASIN (o ficheiro não existe no S3)."
    elif serving_invalid:
        out["warning"] = "O ficheiro de SERVING existe, mas está inválido (vazio ou JSON inválido)."

    return out


# ---------- eval ----------
def _list_eval_part_keys(prefix: str) -> list[str]:
    keys = list_keys(S3_BUCKET, prefix) or []
    keys = [k for k in keys if not k.endswith(".crc")]
    part_keys = [k for k in keys if ("part-" in k) or k.endswith(".json") or k.endswith(".jsonl")]
    return sorted(part_keys)


def load_eval_metrics() -> dict | None:
    try:
        part_keys = _list_eval_part_keys(EVAL_PREFIX_MAIN)
        if not part_keys:
            part_keys = _list_eval_part_keys(EVAL_PREFIX_LEGACY)

        if not part_keys:
            return None

        part = part_keys[-1]
        txt = (read_text_from_s3(S3_BUCKET, part) or "").strip()
        if not txt:
            return None

        try:
            return json.loads(txt)
        except Exception:
            pass

        last_obj = None
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                last_obj = json.loads(line)
            except Exception:
                continue
        return last_obj
    except Exception:
        return None


_eval_cache = {"ts": 0.0, "data": None}
EVAL_CACHE_TTL_SECONDS = 60.0


def load_eval_metrics_cached() -> dict | None:
    now = time.time()
    if _eval_cache["data"] is not None and (now - _eval_cache["ts"]) < EVAL_CACHE_TTL_SECONDS:
        return _eval_cache["data"]
    m = load_eval_metrics()
    _eval_cache["ts"] = now
    _eval_cache["data"] = m
    return m


@app.get("/eval")
def get_eval():
    global EVAL_LOCK_TS

    m = load_eval_metrics_cached()
    if m:
        return m

    now = time.time()
    can_fire = (now - EVAL_LOCK_TS) > EVAL_LOCK_SECONDS

    fired = False
    log_path = None

    if can_fire:
        try:
            os.makedirs("logs/eval_jobs", exist_ok=True)
            log_path = os.path.join("logs", "eval_jobs", "evaluate_models.log")

            env = _BASE_ENV.copy()
            py = sys.executable
            cmd = [py, "-m", "jobs.evaluate_models"]

            with open(log_path, "a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n--- fired {datetime.now().isoformat()} ---\n")
                log.write("cmd=" + " ".join(cmd) + "\n")
                p = _popen_detached_no_window(cmd, log=log, env=env)
                log.write(f"pid={p.pid}\n")

            EVAL_LOCK_TS = now
            fired = True
        except Exception:
            fired = False

    return {
        "error": "Ainda não existem métricas batch (a gerar).",
        "job_fired": fired,
        "log_path": log_path,
        "note": "O backend tentou lançar jobs.evaluate_models automaticamente. Recarrega daqui a 1-2 minutos.",
    }
