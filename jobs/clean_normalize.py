import os
from pyspark.sql import SparkSession, functions as F, types as T
from libs.io_utils import load_config
from libs.text_cleaning import normalize_text
from libs.lang_detect import detect_lang

BASE_PATH = os.getenv("LABGDD_BASE_PATH", "data")
cfg = load_config("conf/pipeline.yaml")
BASE_PATH = cfg["base_path"]
bronze_path = os.path.join(BASE_PATH, cfg["paths"]["bronze"])
silver_path = os.path.join(BASE_PATH, cfg["paths"]["silver"])
min_len = cfg["min_doc_length"]
langs = cfg["languages"]

spark = (
    SparkSession.builder
    .appName("labgdd-clean-normalize")
    .getOrCreate()
)

bronze_path = os.path.join(BASE_PATH, "bronze", "reviews")
silver_path = os.path.join(BASE_PATH, "silver", "reviews")

df = spark.read.parquet(bronze_path)

udf_lang = F.udf(detect_lang, T.StringType())
udf_norm = F.udf(normalize_text, T.StringType())

clean = (
    df
    .withColumn("lang", udf_lang(F.col("review_text")))
    .filter(F.col("lang").isin(*langs))
    .withColumn("review_text_norm", udf_norm(F.col("review_text")))
    .withColumn("doc_len", F.length("review_text_norm"))
    .filter(F.col("doc_len") > F.lit(min_len))
)

clean.write.mode("overwrite").parquet(silver_path)

spark.stop()
