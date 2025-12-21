# jobs/clean_normalize.py
import os
from pyspark.sql import functions as F

from libs.io_utils import load_config
from libs.spark_utils import build_spark

cfg = load_config("conf/pipeline.yaml")
BASE_PATH = cfg["base_path"].rstrip("/")
bronze_path = f"{BASE_PATH}/{cfg['paths']['bronze']}"
silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"

min_len = int(cfg.get("min_doc_length", 50))
target_lang = cfg.get("target_lang", "en")
NUM_PARTS = int(os.getenv("LABGDD_PARTS", "8"))

spark = build_spark("labgdd-clean-normalize")

spark.conf.set("spark.sql.shuffle.partitions", str(NUM_PARTS))
spark.conf.set("spark.default.parallelism", str(NUM_PARTS))

print("🔵 A ler Bronze:", bronze_path)
df = spark.read.parquet(bronze_path)

col_text = "text_norm" if "text_norm" in df.columns else "review_text"

clean = df.filter(F.col(col_text).isNotNull())
clean = clean.withColumn("doc_len", F.length(F.col(col_text)))
clean = clean.filter(F.col("doc_len") > F.lit(min_len))

if "lang" in clean.columns and target_lang and target_lang.lower() != "all":
    clean = clean.filter(F.col("lang") == target_lang)

clean = clean.coalesce(NUM_PARTS)

print("🔵 A gravar Silver:", silver_path)
clean.write.mode("overwrite").parquet(silver_path)

spark.stop()
print("✅ Silver concluído.")
