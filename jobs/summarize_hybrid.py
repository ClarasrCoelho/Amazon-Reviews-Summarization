import os
import sys
import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

from libs.io_utils import load_config
from libs.summarize import summarize_hybrid


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

cfg = load_config("conf/pipeline.yaml")
BASE_PATH = cfg["base_path"].rstrip("/")

silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
summaries_path = f"{BASE_PATH}/{gold_root}/summaries"

MAX_CHARS = 200_000

spark = (
    SparkSession.builder
    .appName("labgdd-summarize-hybrid")
    .master("local[*]")
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.profile.ProfileCredentialsProvider",
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")
    .getOrCreate()
)

print("🔵 A ler Silver de:", silver_path)
df = spark.read.parquet(silver_path)

# 🔒 truncate ANTES do pandas UDF
df = df.withColumn(
    "text_norm",
    F.expr(f"substring(text_norm, 1, {MAX_CHARS})")
)

@pandas_udf(StringType())
def summary_udf(col: pd.Series) -> pd.Series:
    return col.fillna("").apply(
        lambda t: summarize_hybrid(
            t,
            num_sentences_textrank=5,
            max_new_tokens=128,
        )
    )

summarized = (
    df.groupBy("asin")
      .agg(F.collect_list("text_norm").alias("reviews"))
      .withColumn("concat_reviews", F.array_join(F.col("reviews"), " "))
      .withColumn(
          "concat_reviews",
          F.expr(f"substring(concat_reviews, 1, {MAX_CHARS})")
      )
      .withColumn("summary", summary_udf(F.col("concat_reviews")))
      .select("asin", "summary")
)

print("🔵 A gravar Summaries em:", summaries_path)
summarized.write.mode("overwrite").parquet(summaries_path)

spark.stop()
print("✅ Gold/Summaries no S3 concluído.")
