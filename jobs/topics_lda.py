# jobs/topics_lda.py
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA

from libs.io_utils import load_config
from libs.spark_utils import build_spark

cfg = load_config("conf/pipeline.yaml")
BASE_PATH = cfg["base_path"].rstrip("/")
silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
topics_path = f"{BASE_PATH}/{gold_root}/topics"

spark = build_spark("labgdd-topics-lda")

print("🔵 A ler Silver:", silver_path)
df = spark.read.parquet(silver_path)

TEXT_COL = "text_norm"
df = df.filter(df[TEXT_COL].isNotNull())

tokenizer = RegexTokenizer(inputCol=TEXT_COL, outputCol="words", pattern="\\W+")
rem = StopWordsRemover(inputCol="words", outputCol="filtered")

cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=20000, minDF=5)

df_tok = tokenizer.transform(df)
df_filt = rem.transform(df_tok)
cv_model = cv.fit(df_filt)
df_feat = cv_model.transform(df_filt)

lda = LDA(k=10, maxIter=20, featuresCol="features")
lda_model = lda.fit(df_feat)

cols = ["asin", "topicDistribution"]
if "review_id" in df_feat.columns:
    cols.append("review_id")

df_topics = lda_model.transform(df_feat).select(*cols)

print("🔵 A gravar Topics:", topics_path)
df_topics.write.mode("overwrite").parquet(topics_path)

spark.stop()
print("✅ Gold/Topics concluído.")
