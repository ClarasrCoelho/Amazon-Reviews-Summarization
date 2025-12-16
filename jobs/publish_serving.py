import os
import sys
import yaml

from pyspark.sql import SparkSession, functions as F


# -------------------------
# Windows / Hadoop winutils
# -------------------------
HADOOP_HOME = os.getenv("HADOOP_HOME", r"C:\hadoop")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME
os.environ["PATH"] = os.path.join(HADOOP_HOME, "bin") + ";" + os.environ.get("PATH", "")

# garantir python da venv
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")
gold_root = cfg["paths"].get("gold", "gold")

sentiment_path = f"{BASE_PATH}/{gold_root}/sentiment"
absa_path = f"{BASE_PATH}/{gold_root}/absa"
summaries_path = f"{BASE_PATH}/{gold_root}/summaries"
serving_path = f"{BASE_PATH}/{gold_root}/serving"

print("🔵 Sentiment: ", sentiment_path)
print("🔵 ABSA:      ", absa_path)
print("🔵 Summaries: ", summaries_path)
print("🔵 Serving:   ", serving_path)

spark = (
    SparkSession.builder
    .appName("labgdd-publish-serving")
    .master("local[*]")
    .config("spark.driver.extraJavaOptions", f'-Dhadoop.home.dir="{HADOOP_HOME}"')
    .config("spark.executor.extraJavaOptions", f'-Dhadoop.home.dir="{HADOOP_HOME}"')
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.profile.ProfileCredentialsProvider")
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")

    # ---- memória / shuffles (evita OOM) ----
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.maxResultSize", "1g")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df_sent = spark.read.parquet(sentiment_path)
df_absa = spark.read.parquet(absa_path)
df_sum = spark.read.parquet(summaries_path)

# summary column pode chamar-se "summary" ou outra — tenta detectar
summary_col = "summary" if "summary" in df_sum.columns else ("review_summary" if "review_summary" in df_sum.columns else None)
if summary_col is None:
    raise RuntimeError("Não encontro coluna de resumo no gold/summaries (summary/review_summary).")

# 1) Agregar tudo ANTES (reduz volume brutal)
sent_by_asin = (
    df_sent
    .groupBy("asin")
    .agg(F.avg("sentiment_score").alias("avg_sentiment"))
)

# ABSA: evitar collect_set+flatten em massa -> explode, distinct e limita
# (limite 50 aspetos por asin — ajusta se quiseres)
absa_exploded = df_absa.select("asin", "avg_aspect_polarity", F.explode_outer("aspects").alias("aspect"))
absa_by_asin = (
    absa_exploded
    .groupBy("asin")
    .agg(
        F.avg("avg_aspect_polarity").alias("avg_aspect_polarity"),
        F.slice(F.array_sort(F.collect_set("aspect")), 1, 50).alias("all_aspects")
    )
)

sum_by_asin = (
    df_sum
    .groupBy("asin")
    .agg(F.first(F.col(summary_col), ignorenulls=True).alias("summary"))
)

# 2) Joins entre tabelas já pequenas (muito mais leve)
df_serving = (
    sum_by_asin
    .join(sent_by_asin, on="asin", how="left")
    .join(absa_by_asin, on="asin", how="left")
    .select("asin", "summary", "avg_sentiment", "all_aspects", "avg_aspect_polarity")
)

df_serving.write.mode("overwrite").parquet(serving_path)

spark.stop()
print("✅ Gold/serving no S3 concluído.")
