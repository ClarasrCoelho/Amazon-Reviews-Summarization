# jobs/summarize_hybrid.py
import os
import yaml

from pyspark.sql import functions as F
from pyspark.sql import Row

from libs.spark_utils import build_spark
from libs.summarize import summarize_hybrid


def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------
# Config
# ---------------------------
cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")
silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
summaries_root = f"{BASE_PATH}/{gold_root}/summaries"

MAX_CHARS = int(cfg.get("max_chars_summary_input", 8000))
MAX_REVIEWS_PER_ASIN = int(cfg.get("max_reviews_per_asin", 20))

SHARDS = int(cfg.get("summarize_shards", 10))
SHARD_ID = int(os.environ.get("SHARD_ID", cfg.get("summarize_shard_id", 0)))

BATCH_SIZE = int(cfg.get("summarize_batch_size", 200))  # escreve de 200 em 200

if SHARD_ID < 0 or SHARD_ID >= SHARDS:
    raise ValueError(f"SHARD_ID inválido: {SHARD_ID} (0..{SHARDS-1})")

out_path = f"{summaries_root}/shards={SHARDS}/shard_id={SHARD_ID}"

# ---------------------------
# Spark
# ---------------------------
spark = build_spark("labgdd-summarize-hybrid-driver")

# Evita o Spark tentar “otimizar” e baralhar-te o comportamento em local
spark.conf.set("spark.sql.adaptive.enabled", "false")

print("🔵 A ler Silver:", silver_path)
df = spark.read.parquet(silver_path)

if "text_norm" not in df.columns:
    raise RuntimeError("Silver não tem 'text_norm'.")

# mantém só o que interessa
df = (
    df.select("asin", "text_norm")
      .where(F.col("asin").isNotNull())
      .where(F.length(F.col("asin")) > 0)
      .where(F.col("text_norm").isNotNull())
)

# corta cedo
df = df.withColumn("text_norm", F.expr(f"substring(text_norm, 1, {MAX_CHARS})"))

# shard por asin
df = df.where(F.pmod(F.hash(F.col("asin")), F.lit(SHARDS)) == F.lit(SHARD_ID))
print(f"🔵 Shard {SHARD_ID}/{SHARDS} selecionado (asin hash mod {SHARDS} == {SHARD_ID})")

# limita reviews por asin e concatena
grouped = (
    df.groupBy("asin")
      .agg(F.slice(F.collect_list("text_norm"), 1, MAX_REVIEWS_PER_ASIN).alias("reviews"))
      .withColumn("concat_reviews", F.array_join(F.col("reviews"), " "))
      .drop("reviews")
      .withColumn("concat_reviews", F.expr(f"substring(concat_reviews, 1, {MAX_CHARS})"))
      .select("asin", "concat_reviews")
)

# escreve “limpo” para este shard (para poderes repetir shard sem lixo)
print("🧹 A limpar output do shard:", out_path)
spark._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
# overwrite do diretório do shard
grouped.limit(0).selectExpr("asin", "cast(null as string) as summary").write.mode("overwrite").parquet(out_path)

# ---------------------------
# Warm-up do modelo (muito importante)
# ---------------------------
print("🔥 Warm-up do modelo…")
_ = summarize_hybrid("warmup", max_new_tokens=8)

# ---------------------------
# Processar no driver com batches
# ---------------------------
print("🔵 A sumarizar no driver (com batches)…")

buffer = []
count = 0
written = 0

def flush_batch(rows):
    global written
    if not rows:
        return
    sdf = spark.createDataFrame(rows)  # schema inferido: asin, summary
    sdf.write.mode("append").parquet(out_path)
    written += len(rows)
    print(f"✅ Batch escrito: +{len(rows)} (total escrito: {written})")

for r in grouped.toLocalIterator():
    asin = r["asin"]
    text = (r["concat_reviews"] or "").strip()

    if not text:
        summary = ""
    else:
        try:
            summary = summarize_hybrid(text, max_new_tokens=64)
        except Exception:
            summary = text[:800]  # fallback

    buffer.append(Row(asin=asin, summary=summary))
    count += 1

    # prints de progresso reais
    if count % 25 == 0:
        print(f"… processados {count} ASINs (ainda não escritos: {len(buffer)})")

    if len(buffer) >= BATCH_SIZE:
        flush_batch(buffer)
        buffer = []

# flush final
flush_batch(buffer)

spark.stop()
print(f"🏁 Shard {SHARD_ID} concluído. Total processado: {count}, total escrito: {written}")
