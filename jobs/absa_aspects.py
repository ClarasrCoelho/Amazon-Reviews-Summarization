import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType, StructType, StructField
from libs.nlp_udfs import extract_aspects_batch, simple_aspect_polarity

BASE_PATH = os.getenv("LABGDD_BASE_PATH", "data")

spark = (
    SparkSession.builder
    .appName("labgdd-04-absa-aspects")
    .getOrCreate()
)

silver_path = os.path.join(BASE_PATH, "silver", "reviews")
absa_path = os.path.join(BASE_PATH, "gold", "absa")

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

df_absa.write.mode("overwrite").parquet(absa_path)

spark.stop()
