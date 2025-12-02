import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from libs.io_utils import load_config
from libs.text_cleaning import normalize_text
from libs.nlp_udfs import lang_detect_udf, sentence_split_udf

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

cfg = load_config("conf/pipeline.yaml")

BASE_PATH = cfg["base_path"].rstrip("/")

raw_path = f"{BASE_PATH}/{cfg['paths']['raw']}"
bronze_path = f"{BASE_PATH}/{cfg['paths']['bronze']}"

norm_udf = udf(normalize_text, StringType())

spark = (
    SparkSession.builder
    .appName("labgdd-ingest-batch")
    .master("local[*]")
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.profile.ProfileCredentialsProvider"
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")
    .getOrCreate()
)

print("🔵 A ler dados RAW de:", raw_path)
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

print("🔵 A gravar Bronze em:", bronze_path)
df.write.mode("overwrite").parquet(bronze_path)

spark.stop()
print("✅ Ingestão para Bronze (S3) concluída.")
