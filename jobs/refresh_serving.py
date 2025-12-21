# jobs/refresh_serving.py
import os
import json
import time
import yaml
from datetime import datetime, timezone

from pyspark.sql import functions as F

from libs.spark_utils import build_spark
from serving_api.s3_utils import write_text_to_s3, read_text_from_s3

def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    BASE_PATH = cfg["base_path"].rstrip("/")
    gold_root = cfg["paths"].get("gold", "gold")

    asin = os.getenv("SERVING_ASIN", "").strip()
    if not asin:
        raise RuntimeError("Falta SERVING_ASIN no environment (ex: set SERVING_ASIN=B00...).")

    # paths (parquet gold)
    sentiment_path = f"{BASE_PATH}/{gold_root}/sentiment"
    absa_path = f"{BASE_PATH}/{gold_root}/absa"
    summaries_path = f"{BASE_PATH}/{gold_root}/summaries"

    # live user reviews (jsonl por asin)
    user_reviews_path = f"{BASE_PATH}/{gold_root}/user_reviews/{asin}.jsonl"

    # output esperado pela API:
    # s3://labgdd-tp2-2/labgdd/gold/serving/books/{asin}.json
    # A API constrói bucket/prefix a partir de env, por isso usamos esses.
    bucket = os.getenv("S3_BUCKET", "labgdd-tp2-2")
    prefix = os.getenv("S3_PREFIX", "labgdd").strip("/")
    out_key = f"{prefix}/gold/serving/books/{asin}.json"

    spark = build_spark("labgdd-refresh-serving")

    # --- carregar gold batch ---
    df_sent = spark.read.parquet(sentiment_path).filter(F.col("asin") == asin)
    df_absa = spark.read.parquet(absa_path).filter(F.col("asin") == asin)
    df_sum  = spark.read.parquet(summaries_path).filter(F.col("asin") == asin)

    # --- sumarização batch ---
    summary_col = "summary" if "summary" in df_sum.columns else ("review_summary" if "review_summary" in df_sum.columns else None)
    batch_summary = None
    if summary_col:
        row = df_sum.select(F.col(summary_col).alias("s")).limit(1).collect()
        if row:
            batch_summary = row[0]["s"]

    # --- sentiment batch ---
    avg_sent = None
    if df_sent.columns:
        r = df_sent.select(F.avg("sentiment_score").alias("avg")).collect()
        if r:
            avg_sent = r[0]["avg"]

    # --- absa batch ---
    avg_aspect = None
    all_aspects = []
    if df_absa.columns:
        # tenta apanhar aspetos e polaridade
        cols = df_absa.columns
        if "avg_aspect_polarity" in cols:
            r = df_absa.select(F.avg("avg_aspect_polarity").alias("avg")).collect()
            if r:
                avg_aspect = r[0]["avg"]

        if "aspects" in cols:
            asp = (
                df_absa
                .select(F.explode_outer("aspects").alias("a"))
                .groupBy()
                .agg(F.collect_set("a").alias("aspects"))
                .collect()
            )
            if asp and asp[0]["aspects"]:
                all_aspects = sorted([x for x in asp[0]["aspects"] if x])[:50]

    # --- live user reviews -> source_text ---
    source_text = ""
    try:
        # lê o jsonl diretamente do S3 para compor source_text
        txt = read_text_from_s3(bucket, f"{prefix}/gold/user_reviews/{asin}.jsonl")
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        texts = []
        for l in lines[-50:]:
            try:
                obj = json.loads(l)
                t = (obj.get("text") or "").strip()
                if t:
                    texts.append(t)
            except Exception:
                pass
        source_text = " ".join(texts)
    except Exception:
        source_text = ""

    # regra simples: se tiver live text, usa isso como “final summary” (placeholder),
    # senão usa batch_summary. (podes depois trocar por modelo de sumarização live)
    final_summary = None
    if source_text:
        # baseline: devolve 1-2 frases iniciais como “resumo live”
        final_summary = (source_text[:600] + "…") if len(source_text) > 600 else source_text
    else:
        final_summary = batch_summary

    payload = {
        "asin": asin,
        "summary": final_summary,
        "batch_summary": batch_summary,
        "avg_sentiment": float(avg_sent) if avg_sent is not None else None,
        "avg_aspect_polarity": float(avg_aspect) if avg_aspect is not None else None,
        "all_aspects": all_aspects,
        "source_text": source_text,
        "updated_ts": time.time(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    write_text_to_s3(bucket, out_key, json.dumps(payload, ensure_ascii=False))

    spark.stop()
    print(f"✅ serving atualizado: s3://{bucket}/{out_key}")

if __name__ == "__main__":
    main()
