# jobs/build_books_index.py
import yaml
import json
from pyspark.sql import functions as F

from libs.spark_utils import build_spark

def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")
gold_root = cfg["paths"].get("gold", "gold")

books_meta_path = f"{BASE_PATH}/{gold_root}/books_meta"
serving_path = f"{BASE_PATH}/{gold_root}/serving"

# destino esperado pela tua app (pelo erro no UI)
# gold/serving/books/{asin}.json
books_json_out = f"{BASE_PATH}/{gold_root}/serving/books"

spark = build_spark("labgdd-build-books-index")

print("🔵 A ler Gold/books_meta:", books_meta_path)
meta = spark.read.parquet(books_meta_path)

print("🔵 A ler Gold/serving:", serving_path)
serv = spark.read.parquet(serving_path)

df = meta.join(serv, on="asin", how="left")

# garantir colunas mínimas
for c in ["avg_rating", "n_reviews"]:
    if c not in df.columns:
        df = df.withColumn(c, F.lit(None))
for c in ["summary", "avg_sentiment", "all_aspects", "avg_aspect_polarity"]:
    if c not in df.columns:
        df = df.withColumn(c, F.lit(None))

# cria 1 JSON por ASIN
df_json = df.select(
    "asin",
    F.to_json(
        F.struct(
            F.col("asin"),
            F.col("avg_rating"),
            F.col("n_reviews"),
            F.col("summary"),
            F.col("avg_sentiment"),
            F.col("all_aspects"),
            F.col("avg_aspect_polarity"),
        )
    ).alias("json")
)

# escrever como “files” por ASIN:
# truque: escrever partição por asin e depois renomear não dá no S3 facilmente,
# então escrevemos um ficheiro por asin usando repartition + write text com path.
# (Funciona bem em s3a: cada asin fica numa pasta com part-0000...json)
print("🔵 A escrever JSON por ASIN em:", books_json_out)
(
    df_json
    .repartition("asin")
    .write
    .mode("overwrite")
    .partitionBy("asin")
    .text(books_json_out)
)

spark.stop()
print("✅ serving/books/{asin} criado (em pastas por asin).")
