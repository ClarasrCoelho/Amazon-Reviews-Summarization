import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
from libs.summarize import summarize_hybrid

BASE_PATH = os.getenv("LABGDD_BASE_PATH", "data")

spark = (
    SparkSession.builder
    .appName("labgdd-summarize-hybrid")
    .getOrCreate()
)

silver_path = os.path.join(BASE_PATH, "silver", "reviews")
summaries_path = os.path.join(BASE_PATH, "gold", "summaries")

df = spark.read.parquet(silver_path)

@pandas_udf(StringType())
def summary_udf(col: pd.Series) -> pd.Series:
    return col.fillna("").apply(
        lambda t: summarize_hybrid(t, num_sentences_textrank=5,
                                   max_new_tokens=128)
    )

summarized = df.groupBy("asin").agg(
    F.collect_list("review_text_norm").alias("reviews")
).withColumn(
    "concat_reviews", F.array_join(F.col("reviews"), " ")
).withColumn(
    "summary", summary_udf(F.col("concat_reviews"))
).select(
    "asin",
    "summary"
)

summarized.write.mode("overwrite").parquet(summaries_path)

spark.stop()
