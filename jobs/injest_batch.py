# jobs/injest_batch.py
import sys
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from libs.io_utils import load_config
from libs.text_cleaning import normalize_text
from libs.nlp_udfs import lang_detect_udf, sentence_split_udf
from libs.spark_utils import build_spark

cfg = load_config("conf/pipeline.yaml")
BASE_PATH = cfg["base_path"].rstrip("/")

raw_path = f"{BASE_PATH}/{cfg['paths']['raw']}"
bronze_path = f"{BASE_PATH}/{cfg['paths']['bronze']}"

norm_udf = udf(normalize_text, StringType())

spark = build_spark("labgdd-ingest-batch")

print("🔵 A ler RAW:", raw_path)
df = spark.read.json(raw_path)

df = (
    df.withColumnRenamed("reviewText", "review_text")
      .withColumnRenamed("summary", "review_summary")
)

print("🔵 A normalizar texto...")
df = df.withColumn("text_norm", norm_udf(col("review_text")))

print("🔵 A detectar língua...")
df = df.withColumn("lang", lang_detect_udf(col("text_norm")))

print("🔵 A dividir em frases...")
df = df.withColumn("sentences", sentence_split_udf(col("text_norm")))

print("🔵 A gravar Bronze:", bronze_path)
df.write.mode("overwrite").parquet(bronze_path)

spark.stop()
print("✅ Bronze concluído.")
