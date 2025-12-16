import os
import pandas as pd
import yaml

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

from libs.nlp_udfs import predict_sentiment_batch


# -------- 1) Ler config.yaml --------

def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()

BASE_PATH = cfg.get("base_path", "data")
PATHS = cfg.get("paths", {})
MIN_DOC_LEN = cfg.get("min_doc_length", 0)
LANGUAGES = cfg.get("languages", [])


def mkpath(*parts):
    """
    Constrói paths que funcionam tanto para disco local como para S3.
    - Se BASE_PATH começar por s3a://, faz concatenação com "/"
    - Caso contrário, usa os.path.join
    """
    if BASE_PATH.startswith("s3a://"):
        return "/".join([BASE_PATH.rstrip("/")] + [p.strip("/") for p in parts])
    return os.path.join(BASE_PATH, *parts)


silver_path = mkpath(PATHS.get("silver", "silver/reviews"))
sentiment_path = mkpath(PATHS.get("gold", "gold"), "sentiment")

print(f"🔵 A ler Silver de: {silver_path}")

# -------- 2) SparkSession com suporte S3A --------


spark = (
    SparkSession.builder
    .appName("labgdd-embeddings-sentiment")
    .master("local[*]")
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.profile.ProfileCredentialsProvider"
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    # 🔴 Aumentar timeouts para não rebentar em 2 minutos
    .config("spark.network.timeout", "600s")             # default ~120s
    .config("spark.executor.heartbeatInterval", "60s")   # manter abaixo do timeout
    .getOrCreate()
)

# -------- 3) Ler Silver --------

df = spark.read.parquet(silver_path)
total_reviews = df.count()
print(f"🔵 Nº total de reviews no Silver (antes de filtros): {total_reviews}")


print("🔵 Schema Silver:")
df.printSchema()

# Filtro por comprimento mínimo
if "doc_len" in df.columns and MIN_DOC_LEN > 0:
    df = df.filter(F.col("doc_len") >= MIN_DOC_LEN)

# Filtro por línguas (se configurado e se existir coluna lang)
if LANGUAGES and "lang" in df.columns:
    df = df.filter(F.col("lang").isin(LANGUAGES))

print(f"🔵 Nº de reviews após filtro: {df.count()}")


# -------- 4) UDF de sentimento (pandas_udf) --------

@pandas_udf(DoubleType())
def sentiment_udf(col: pd.Series) -> pd.Series:
    return predict_sentiment_batch(col)


print("🔵 A aplicar modelo de Sentiment...")
scored = df.withColumn(
    "sentiment_score",
    sentiment_udf(F.col("text_norm"))
)

print(f"🔵 A gravar Sentiment em: {sentiment_path}")
(
    scored.select(
        "reviewerID",  # se tiveres "review_id", troca aqui
        "asin",
        "sentiment_score"
    )
    .write.mode("overwrite")
    .parquet(sentiment_path)
)

print("✅ Job embeddings_sentiment terminado com sucesso.")
spark.stop()
