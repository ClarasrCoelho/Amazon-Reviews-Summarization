# jobs/absa_aspects.py
import yaml
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StringType, DoubleType

from libs.nlp_udfs import extract_aspects_batch, simple_aspect_polarity
from libs.spark_utils import build_spark

def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")
silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
absa_path = f"{BASE_PATH}/{gold_root}/absa"

spark = build_spark("labgdd-absa-aspects")

print("🔵 A ler Silver:", silver_path)
df = spark.read.parquet(silver_path)

text_col = "text_norm" if "text_norm" in df.columns else ("review_text" if "review_text" in df.columns else None)
if text_col is None:
    raise RuntimeError("Não encontro coluna de texto (text_norm/review_text).")

df = df.repartition(4)

@pandas_udf(ArrayType(StringType()))
def aspects_udf(texts: pd.Series) -> pd.Series:
    res = extract_aspects_batch(texts)
    if res is None:
        return pd.Series([[]] * len(texts))
    out = []
    for r in res:
        if r is None:
            out.append([])
        elif isinstance(r, np.ndarray):
            out.append(r.tolist())
        elif isinstance(r, (list, tuple)):
            out.append(list(r))
        else:
            try:
                out.append(list(r))
            except Exception:
                out.append([])
    return pd.Series(out)

@pandas_udf(DoubleType())
def avg_aspect_polarity_udf(texts: pd.Series, aspects: pd.Series) -> pd.Series:
    texts = texts.fillna("").astype(str)

    def normalize_aspects(a):
        if a is None:
            return []
        if isinstance(a, float) and pd.isna(a):
            return []
        if isinstance(a, np.ndarray):
            return a.tolist()
        if isinstance(a, (list, tuple)):
            return list(a)
        try:
            return list(a)
        except Exception:
            return []

    def row_score(t: str, a):
        asp_list = normalize_aspects(a)
        if not asp_list:
            return 0.0
        scores = [simple_aspect_polarity(t, asp) for asp in asp_list if asp]
        return float(sum(scores)) / float(len(scores)) if scores else 0.0

    return pd.Series([row_score(t, a) for t, a in zip(texts.tolist(), aspects.tolist())], dtype="float64")

print("🔵 A extrair aspetos...")
df2 = df.withColumn("aspects", aspects_udf(F.col(text_col)))
print("🔵 A calcular polaridade média...")
df2 = df2.withColumn("avg_aspect_polarity", avg_aspect_polarity_udf(F.col(text_col), F.col("aspects")))

select_cols = [c for c in ["review_id", "reviewerID", "asin"] if c in df2.columns]
df_absa = df2.select(*select_cols, "aspects", "avg_aspect_polarity")

print("🔵 A gravar ABSA:", absa_path)
df_absa.write.mode("overwrite").parquet(absa_path)

spark.stop()
print("✅ Gold/ABSA concluído.")
