# jobs/publish_serving.py
import yaml
from pyspark.sql import functions as F

from libs.spark_utils import build_spark

def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")
gold_root = cfg["paths"].get("gold", "gold")

sentiment_path = f"{BASE_PATH}/{gold_root}/sentiment"
absa_path = f"{BASE_PATH}/{gold_root}/absa"
summaries_path = f"{BASE_PATH}/{gold_root}/summaries"
serving_path = f"{BASE_PATH}/{gold_root}/serving"

print("🔵 Sentiment:", sentiment_path)
print("🔵 ABSA:     ", absa_path)
print("🔵 Summaries:", summaries_path)
print("🔵 Serving:  ", serving_path)

spark = build_spark("labgdd-publish-serving")

df_sent = spark.read.parquet(sentiment_path)
df_absa = spark.read.parquet(absa_path)
df_sum = spark.read.parquet(summaries_path)

summary_col = "summary" if "summary" in df_sum.columns else ("review_summary" if "review_summary" in df_sum.columns else None)
if summary_col is None:
    raise RuntimeError("Não encontro coluna de resumo no gold/summaries (summary/review_summary).")

sent_by_asin = df_sent.groupBy("asin").agg(F.avg("sentiment_score").alias("avg_sentiment"))

absa_exploded = df_absa.select("asin", "avg_aspect_polarity", F.explode_outer("aspects").alias("aspect"))
absa_by_asin = (
    absa_exploded
    .groupBy("asin")
    .agg(
        F.avg("avg_aspect_polarity").alias("avg_aspect_polarity"),
        F.slice(F.array_sort(F.collect_set("aspect")), 1, 50).alias("all_aspects"),
    )
)

sum_by_asin = df_sum.groupBy("asin").agg(F.first(F.col(summary_col), ignorenulls=True).alias("summary"))

df_serving = (
    sum_by_asin
    .join(sent_by_asin, on="asin", how="left")
    .join(absa_by_asin, on="asin", how="left")
    .select("asin", "summary", "avg_sentiment", "all_aspects", "avg_aspect_polarity")
)

df_serving.write.mode("overwrite").parquet(serving_path)

spark.stop()
print("✅ Gold/serving concluído.")
