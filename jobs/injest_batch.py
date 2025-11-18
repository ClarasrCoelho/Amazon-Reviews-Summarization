import os
from pyspark.sql import SparkSession
from libs.io_utils import load_config

cfg = load_config("conf/pipeline.yaml")

BASE_PATH = cfg["base_path"]
raw_path = os.path.join(BASE_PATH, cfg["paths"]["raw"])        # data/raw/Books_sample.json
bronze_path = os.path.join(BASE_PATH, cfg["paths"]["bronze"])  # data/bronze/reviews

print("RAW_PATH    =", raw_path)
print("BRONZE_PATH =", bronze_path)

spark = (
    SparkSession.builder
    .appName("labgdd-ingest-batch")
    .master("local[*]")
    .config("spark.driver.memory", "4g")   # sobe para 6g/8g se tiveres RAM
    .getOrCreate()
)

print("🔵 A ler dados RAW...")
df = spark.read.json(raw_path)

print("🔵 Primeiras linhas:")
df.show(5, truncate=False)

print("🔵 Total de linhas:", df.count())

df = (
    df.withColumnRenamed("reviewText", "review_text")
      .withColumnRenamed("summary", "review_summary")
)

print("🔵 A gravar Bronze...")
df.write.mode("overwrite").parquet(bronze_path)

spark.stop()
print("✅ Ingestão concluída.")
