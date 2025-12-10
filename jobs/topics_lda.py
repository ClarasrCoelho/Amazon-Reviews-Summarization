import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA

from libs.io_utils import load_config

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

cfg = load_config("conf/pipeline.yaml")
BASE_PATH = cfg["base_path"].rstrip("/")

silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
topics_path = f"{BASE_PATH}/{gold_root}/topics"

spark = (
    SparkSession.builder
    .appName("labgdd-topics-lda")
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

tokenizer = Tokenizer(inputCol="review_text_norm", outputCol="words")
rem = StopWordsRemover(inputCol="words", outputCol="filtered")
cv = CountVectorizer(
    inputCol="filtered",
    outputCol="features",
    vocabSize=20000,
    minDF=5,
)

df_tok = tokenizer.transform(df)
df_filt = rem.transform(df_tok)
cv_model = cv.fit(df_filt)
df_feat = cv_model.transform(df_filt)

lda = LDA(k=10, maxIter=20, featuresCol="features")
lda_model = lda.fit(df_feat)

df_topics = lda_model.transform(df_feat).select(
    "review_id",
    "asin",
    "topicDistribution"
)

print("🔵 A gravar Topics em:", topics_path)
df_topics.write.mode("overwrite").parquet(topics_path)

spark.stop()
print("✅ Gold/Topics no S3 concluído.")
