import os
import sys
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType, StructType, StructField

from libs.io_utils import load_config
from libs.nlp_udfs import extract_aspects_batch, simple_aspect_polarity

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

cfg = load_config("conf/pipeline.yaml")
BASE_PATH = cfg["base_path"].rstrip("/")

silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
absa_path = f"{BASE_PATH}/{gold_root}/absa"

spark = (
    SparkSession.builder
    .appName("labgdd-04-absa-aspects")
    .master("local[*]")
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.profile.ProfileCredentialsProvider",
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")
    .getOrCreate()
)

print("🔵 A ler Silver de:", silver_path)
df = spark.read.parquet(silver_path)

@pandas_udf(StringType())
def aspects_udf(col: pd.Series) -> pd.Series:
    return extract_aspects_batch(col)

schema_absa = StructType([
    StructField("aspects", StringType(), True),
    StructField("aspect_sentiments", StringType(), True),
])

@pandas_udf(schema_absa)
def aspect_polarity_udf(texts: pd.Series,
                        aspects: pd.Series) -> pd.DataFrame:
    return simple_aspect_polarity(texts, aspects)

df_with_aspects = df.withColumn(
    "aspects", aspects_udf(F.col("review_text_norm"))
)

df_absa = df_with_aspects.withColumn(
    "absa",
    aspect_polarity_udf(F.col("review_text_norm"), F.col("aspects"))
).select(
    "review_id",
    "asin",
    F.col("absa.aspects").alias("aspects"),
    F.col("absa.aspect_sentiments").alias("aspect_sentiments")
)

print("🔵 A gravar ABSA em:", absa_path)
df_absa.write.mode("overwrite").parquet(absa_path)

spark.stop()
print("✅ Gold/ABSA no S3 concluído.")
