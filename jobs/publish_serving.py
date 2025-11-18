import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_PATH = os.getenv("LABGDD_BASE_PATH", "data")

spark = (
    SparkSession.builder
    .appName("labgdd-publish-serving")
    .getOrCreate()
)

sentiment_path = os.path.join(BASE_PATH, "gold", "sentiment")
absa_path = os.path.join(BASE_PATH, "gold", "absa")
summaries_path = os.path.join(BASE_PATH, "gold", "summaries")
serving_path = os.path.join(BASE_PATH, "gold", "serving")

df_sent = spark.read.parquet(sentiment_path)
df_absa = spark.read.parquet(absa_path)
df_sum = spark.read.parquet(summaries_path)

df_serving = (
    df_sum.alias("s")
    .join(df_sent.alias("t"), on="asin", how="left")
    .join(df_absa.alias("a"), on=["asin"], how="left")
    .groupBy("asin", "summary")
    .agg(
        F.avg("sentiment_score").alias("avg_sentiment"),
        F.collect_set("aspects").alias("all_aspects"),
    )
)

df_serving.write.mode("overwrite").parquet(serving_path)

spark.stop()
