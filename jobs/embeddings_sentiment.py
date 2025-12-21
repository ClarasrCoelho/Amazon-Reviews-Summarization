# jobs/embeddings_sentiment.py
import yaml
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

from libs.spark_utils import build_spark
from libs.nlp_udfs import predict_sentiment_batch

def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")
silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
sentiment_path = f"{BASE_PATH}/{gold_root}/sentiment"

spark = build_spark("labgdd-sentiment")

print("🔵 A ler Silver:", silver_path)
df = spark.read.parquet(silver_path)

text_col = "text_norm" if "text_norm" in df.columns else ("review_text" if "review_text" in df.columns else None)
if text_col is None:
    raise RuntimeError("Não encontro coluna de texto (text_norm/review_text).")

@pandas_udf(DoubleType())
def sentiment_udf(texts: pd.Series) -> pd.Series:
    return predict_sentiment_batch(texts).astype("float64")

print("🔵 A calcular sentiment...")
df_out = df.withColumn("sentiment_score", sentiment_udf(F.col(text_col)))

keep = [c for c in ["review_id", "reviewerID", "asin", "overall", "unixReviewTime", "reviewTime"] if c in df_out.columns]
df_out = df_out.select(*keep, "sentiment_score")

print("🔵 A gravar Gold/Sentiment:", sentiment_path)
df_out.write.mode("overwrite").parquet(sentiment_path)

spark.stop()
print("✅ Gold/Sentiment concluído.")
