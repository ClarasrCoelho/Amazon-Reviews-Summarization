# jobs/build_books_meta.py
import yaml
from pyspark.sql import functions as F

from libs.spark_utils import build_spark

def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")
silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
books_meta_path = f"{BASE_PATH}/{gold_root}/books_meta"

spark = build_spark("labgdd-build-books-meta")

print("🔵 A ler Silver:", silver_path)
df = spark.read.parquet(silver_path)

if "asin" not in df.columns:
    raise RuntimeError("Silver não tem coluna 'asin'.")
if "overall" not in df.columns:
    # se o dataset usar outro nome, ajusta aqui
    raise RuntimeError("Silver não tem coluna 'overall' (rating).")

meta = (
    df.groupBy("asin")
      .agg(
          F.count("*").alias("n_reviews"),
          F.avg("overall").alias("avg_rating"),
      )
)

print("🔵 A gravar Gold/books_meta:", books_meta_path)
meta.write.mode("overwrite").parquet(books_meta_path)

spark.stop()
print("✅ Gold/books_meta concluído.")
