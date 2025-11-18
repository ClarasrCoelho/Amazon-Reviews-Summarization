import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA

BASE_PATH = os.getenv("LABGDD_BASE_PATH", "data")

spark = (
    SparkSession.builder
    .appName("labgdd-topics-lda")
    .getOrCreate()
)

silver_path = os.path.join(BASE_PATH, "silver", "reviews")
topics_path = os.path.join(BASE_PATH, "gold", "topics")

df = spark.read.parquet(silver_path)

tokenizer = Tokenizer(inputCol="review_text_norm", outputCol="words")
rem = StopWordsRemover(inputCol="words", outputCol="filtered")
cv = CountVectorizer(inputCol="filtered", outputCol="features",
                     vocabSize=20000, minDF=5)

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

df_topics.write.mode("overwrite").parquet(topics_path)

spark.stop()
