import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from libs.nlp_udfs import predict_sentiment_batch

BASE_PATH = os.getenv("LABGDD_BASE_PATH", "data")

spark = (
    SparkSession.builder
    .appName("labgdd-03-embeddings-sentiment")
    .getOrCreate()
)

silver_path = os.path.join(BASE_PATH, "silver", "reviews")
sentiment_path = os.path.join(BASE_PATH, "gold", "sentiment")

df = spark.read.parquet(silver_path)

@pandas_udf(DoubleType())
def sentiment_udf(col: pd.Series) -> pd.Series:
    return predict_sentiment_batch(col)

scored = df.withColumn("sentiment_score",
                       sentiment_udf(F.col("review_text_norm")))

scored.select(
    "review_id",
    "asin",
    "sentiment_score"
).write.mode("overwrite").parquet(sentiment_path)

spark.stop()
