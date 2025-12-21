import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone, timedelta

# UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import subprocess

import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType

from libs.io_utils import load_config
from libs.text_cleaning import normalize_text
from libs.nlp_udfs import lang_detect_udf, sentence_split_udf

from serving_api.s3_utils import write_text_to_s3, read_text_from_s3
from libs.summarize import run_summary_models


S3_BUCKET = os.getenv("S3_BUCKET", "labgdd-tp2-2")
S3_PREFIX = os.getenv("S3_PREFIX", "labgdd").strip("/")

BRONZE_EVENTS_PREFIX = f"s3a://{S3_BUCKET}/{S3_PREFIX}/bronze/reviews_events/"
SILVER_BATCH_PREFIX = None  # vem do pipeline.yaml
SILVER_USER_PREFIX = f"s3a://{S3_BUCKET}/{S3_PREFIX}/silver/user_reviews"   # incremental
SERVING_BOOK_PREFIX = f"{S3_PREFIX}/gold/serving/books/"                   # write_text_to_s3 usa bucket+key

MAX_CHARS = 200_000

# -------------------------------------------------------------------
# PATCH WINDOWS: impedir PySpark de abrir janela CMD ao lançar JVM
# -------------------------------------------------------------------
_PYSPARK_PATCHED = False

def _patch_pyspark_gateway_no_window():
    global _PYSPARK_PATCHED
    if _PYSPARK_PATCHED:
        return
    if os.name != "nt":
        _PYSPARK_PATCHED = True
        return

    try:
        import pyspark.java_gateway as jg

        orig_popen = jg.Popen

        def popen_no_window(*args, **kwargs):
            # força sem janela
            kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)

            si = kwargs.get("startupinfo")
            if si is None:
                si = subprocess.STARTUPINFO()
                kwargs["startupinfo"] = si

            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0

            # também evita bloquear em handles de stdin
            kwargs.setdefault("stdin", subprocess.DEVNULL)

            return orig_popen(*args, **kwargs)

        jg.Popen = popen_no_window
        _PYSPARK_PATCHED = True

    except Exception:
        # se falhar o patch, não crasha — mas pode voltar a abrir janela
        _PYSPARK_PATCHED = True


def serving_book_key(asin: str) -> str:
    return f"{SERVING_BOOK_PREFIX}{asin}.json"


def try_load_serving(asin: str):
    try:
        txt = read_text_from_s3(S3_BUCKET, serving_book_key(asin))
        return json.loads(txt)
    except Exception:
        return None


def write_serving(asin: str, payload: dict):
    write_text_to_s3(
        bucket=S3_BUCKET,
        key=serving_book_key(asin),
        text=json.dumps(payload, ensure_ascii=False),
        content_type="application/json",
    )


def build_spark(app_name: str):
    """
    Windows fix: força driver a bindar em 127.0.0.1 para evitar issues
    tipo NPE no BlockManagerMasterEndpoint (local mode).

    + Patch para NÃO abrir CMD ao lançar JVM.
    """
    _patch_pyspark_gateway_no_window()

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")
        # ---- fixes de rede / windows ----
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.ip", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
    )
    return builder.getOrCreate()


def list_recent_partitions(hours_back: int = 6) -> list[str]:
    now = datetime.now(timezone.utc)
    paths = []
    for i in range(hours_back + 1):
        t = now - timedelta(hours=i)
        dt = t.strftime("%Y-%m-%d")
        hh = t.strftime("%H")
        paths.append(f"{BRONZE_EVENTS_PREFIX}dt={dt}/hour={hh}/*.jsonl")
    return paths


def spark_read_jsonl(spark: SparkSession, glob_paths: list[str]):
    df_txt = None
    for p in glob_paths:
        try:
            d = spark.read.text(p)
            df_txt = d if df_txt is None else df_txt.unionByName(d)
        except Exception:
            continue

    if df_txt is None:
        return None

    df = df_txt.select(F.col("value").alias("line")).where(F.col("line").isNotNull())

    df2 = (
        df
        .withColumn("asin", F.get_json_object("line", "$.asin"))
        .withColumn("text", F.get_json_object("line", "$.text"))
        .withColumn("rating", F.get_json_object("line", "$.rating"))
        .withColumn("user", F.get_json_object("line", "$.user"))
        .withColumn("ts", F.get_json_object("line", "$.ts"))
        .withColumn("source", F.get_json_object("line", "$.source"))
        .select("asin", "text", "rating", "user", "ts", "source")
    )
    return df2


def recompute_serving_for_asin(spark: SparkSession, asin: str, silver_batch_path: str, note_extra: str | None = None):
    # silver batch
    texts_b = []
    try:
        df_b = spark.read.parquet(silver_batch_path).where(F.col("asin") == asin)
        text_col_b = "text_norm" if "text_norm" in df_b.columns else ("review_text" if "review_text" in df_b.columns else None)
        if text_col_b:
            texts_b = [r[0] for r in df_b.select(F.col(text_col_b)).where(F.col(text_col_b).isNotNull()).limit(200).collect()]
    except Exception:
        texts_b = []

    # silver incremental
    texts_u = []
    try:
        df_u = spark.read.parquet(SILVER_USER_PREFIX).where(F.col("asin") == asin)
        text_col_u = "text_norm" if "text_norm" in df_u.columns else ("review_text" if "review_text" in df_u.columns else None)
        if text_col_u:
            texts_u = [r[0] for r in df_u.select(F.col(text_col_u)).where(F.col(text_col_u).isNotNull()).limit(200).collect()]
    except Exception:
        texts_u = []

    texts = [t for t in (texts_b + texts_u) if isinstance(t, str) and t.strip()]
    concat = " ".join(texts).strip()
    if len(concat) > MAX_CHARS:
        concat = concat[:MAX_CHARS]

    best_payload = {"name": "none", "summary": "", "score": 0.0, "reasons": ["no_text"]}
    if concat:
        try:
            out = run_summary_models(concat)
            b = out["best"]
            best_payload = {"name": b.name, "summary": b.summary, "score": float(b.score), "reasons": b.reasons}
        except Exception as e:
            best_payload = {"name": "failed", "summary": "", "score": 0.0, "reasons": [f"summary_failed:{type(e).__name__}"]}

    payload = {
        "asin": asin,
        "updated_ts": time.time(),
        "source_text": concat,
        "summary": best_payload["summary"],
        "model": best_payload["name"],
        "model_score": best_payload["score"],
        "model_reasons": best_payload["reasons"],
        "note": "Serving recalculado pelo job process_new_reviews (batch-like).",
    }
    if note_extra:
        payload["note_extra"] = note_extra

    write_serving(asin, payload)
    return payload


def main():
    # garante python da venv para driver/executors
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    p = argparse.ArgumentParser()
    p.add_argument("--asin", default="", help="ASIN opcional (se vazio, processa todos os que aparecerem)")
    p.add_argument("--hours_back", type=int, default=int(os.getenv("LABGDD_HOURS_BACK", "6")))
    p.add_argument("--min_new_reviews", type=int, default=int(os.getenv("LABGDD_MIN_NEW_REVIEWS", "3")))
    args = p.parse_args()

    asin_arg = args.asin.strip() or ""

    cfg = load_config("conf/pipeline.yaml")
    base = cfg["base_path"].rstrip("/")
    silver_batch_path = f"{base}/{cfg['paths']['silver']}"
    global SILVER_BATCH_PREFIX
    SILVER_BATCH_PREFIX = silver_batch_path

    print(f"[process_new_reviews] start asin={asin_arg or 'ALL'} hours_back={args.hours_back} min_new_reviews={args.min_new_reviews}", flush=True)
    print(f"[paths] silver_batch={silver_batch_path}", flush=True)
    print(f"[paths] silver_user={SILVER_USER_PREFIX}", flush=True)
    print(f"[paths] bronze={BRONZE_EVENTS_PREFIX}", flush=True)

    spark = build_spark("labgdd-process-new-reviews")
    spark.sparkContext.setLogLevel("WARN")

    paths = list_recent_partitions(hours_back=args.hours_back)
    df_events = spark_read_jsonl(spark, paths)

    if df_events is None:
        if asin_arg:
            print("[bronze] nenhum evento recente encontrado; vou recalcular serving na mesma (asin explícito).", flush=True)
            recompute_serving_for_asin(
                spark,
                asin_arg,
                silver_batch_path,
                note_extra="No new bronze events in the selected window; recomputed from silver.",
            )
        else:
            print("[bronze] Nenhum evento bronze encontrado nas partições recentes.", flush=True)

        spark.stop()
        return

    df_events = df_events.where(F.col("asin").isNotNull() & F.col("text").isNotNull())

    if asin_arg:
        df_events = df_events.where(F.col("asin") == F.lit(asin_arg))

    norm_udf = F.udf(normalize_text, StringType())
    df_norm = (
        df_events
        .withColumnRenamed("text", "review_text")
        .withColumn("text_norm", norm_udf(F.col("review_text")))
        .withColumn("lang", lang_detect_udf(F.col("text_norm")))
        .withColumn("sentences", sentence_split_udf(F.col("text_norm")))
        .withColumn("doc_len", F.length(F.col("text_norm")))
        .withColumn("ingested_ts", F.lit(time.time()))
    )

    min_len = int(cfg.get("min_doc_length", 50))
    target_lang = cfg.get("target_lang", "en")

    df_clean = df_norm.where(F.col("text_norm").isNotNull())
    df_clean = df_clean.where(F.col("doc_len") > F.lit(min_len))

    if target_lang and target_lang.lower() != "all":
        df_clean = df_clean.where(F.col("lang") == F.lit(target_lang))

    has_clean = df_clean.limit(1).count() > 0

    if not has_clean:
        if asin_arg:
            print("[clean] após filtros não sobrou nada, mas asin é explícito -> recompute serving na mesma.", flush=True)
            recompute_serving_for_asin(
                spark,
                asin_arg,
                silver_batch_path,
                note_extra="New events existed but were filtered out (lang/min_len). Recomputed from silver.",
            )
            spark.stop()
            return

        print("[clean] Eventos existem, mas após normalização/filtros não sobrou nada.", flush=True)
        spark.stop()
        return

    print("[silver] Append silver incremental:", SILVER_USER_PREFIX, flush=True)
    (
        df_clean
        .select(
            F.col("asin").cast("string").alias("asin"),
            F.col("review_text").cast("string").alias("review_text"),
            F.col("text_norm").cast("string").alias("text_norm"),
            F.col("lang").cast("string").alias("lang"),
            F.col("sentences"),
            F.col("doc_len").cast("int").alias("doc_len"),
            F.col("rating").cast("double").alias("overall"),
            F.col("user").cast("string").alias("reviewerID"),
            F.col("ts").cast("double").alias("event_ts"),
            F.col("source").cast("string").alias("source"),
            F.col("ingested_ts").cast("double").alias("ingested_ts"),
        )
        .write.mode("append")
        .parquet(SILVER_USER_PREFIX)
    )

    asin_counts = df_clean.groupBy("asin").agg(F.count(F.lit(1)).alias("new_cnt")).collect()

    todo = []
    for r in asin_counts:
        a = r["asin"]
        c = int(r["new_cnt"] or 0)
        if c >= args.min_new_reviews:
            todo.append(a)

    if asin_arg:
        todo = [asin_arg]

    if not todo:
        print(f"🟡 Não vou recalcular summaries (min_new_reviews={args.min_new_reviews}).", flush=True)
        spark.stop()
        return

    for a in todo:
        try:
            print("[serving] Recalcular:", a, flush=True)
            recompute_serving_for_asin(spark, a, silver_batch_path)
            print("[serving] OK:", a, flush=True)
        except Exception as e:
            print("[serving] FALHOU:", a, type(e).__name__, str(e), flush=True)

    spark.stop()
    print("[process_new_reviews] terminado.", flush=True)


if __name__ == "__main__":
    main()
