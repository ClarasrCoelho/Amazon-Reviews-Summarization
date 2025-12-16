import os
import sys
import yaml

from pyspark.sql import SparkSession, functions as F


def norm_win_path(p: str) -> str:
    """Normaliza paths para o formato mais estável no Java/Hadoop em Windows: C:/..."""
    return os.path.abspath(p).replace("\\", "/")


# --- Hadoop / Spark env (Windows) ---
HADOOP_HOME = norm_win_path(os.getenv("HADOOP_HOME", r"C:\hadoop"))

# força variáveis que o Hadoop/Spark usam em Windows
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME

# garante que o Spark usa o python da venv
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")

silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")

serving_path = f"{BASE_PATH}/{gold_root}/serving"
catalog_out = f"{BASE_PATH}/{gold_root}/catalog/books_index"  # pasta com JSONL part files

# --- Spark session ---
spark = (
    SparkSession.builder
    .appName("labgdd-build-books-index")
    .master("local[*]")
    # hadoop.home.dir no Java (sem aspas e com C:/...)
    .config("spark.driver.extraJavaOptions", f"-Dhadoop.home.dir={HADOOP_HOME}")
    .config("spark.executor.extraJavaOptions", f"-Dhadoop.home.dir={HADOOP_HOME}")
    # se não estiveres a usar S3 nesta execução, podes comentar as 3 configs abaixo
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.profile.ProfileCredentialsProvider",
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("🔵 HADOOP_HOME =", HADOOP_HOME)
print("🔵 A ler Silver:", silver_path)
df_silver = spark.read.parquet(silver_path)

print("🔵 A ler Gold/serving:", serving_path)
df_serv = spark.read.parquet(serving_path)

# Silver: stats por asin
asin_col = "asin" if "asin" in df_silver.columns else None
if asin_col is None:
    raise RuntimeError("Não encontrei coluna 'asin' no silver.")

overall_col = "overall" if "overall" in df_silver.columns else None

agg_exprs = [F.count(F.lit(1)).alias("review_count")]
if overall_col:
    agg_exprs.append(F.avg(F.col(overall_col)).alias("avg_rating"))

df_stats = df_silver.groupBy("asin").agg(*agg_exprs)

# Serving (conforme publish_serving)
df_index = (
    df_serv.select(
        "asin",
        F.col("summary").alias("book_summary"),
        F.col("avg_sentiment").alias("avg_sentiment"),
        F.col("avg_aspect_polarity").alias("avg_aspect_polarity"),
        F.col("all_aspects").alias("all_aspects"),
    )
    .join(df_stats, on="asin", how="left")
    # placeholders (se não tiveres metadata)
    .withColumn("title", F.concat(F.lit("Livro "), F.col("asin")))
    .withColumn("imUrl", F.lit(""))
    .withColumn("price", F.lit(None).cast("double"))
    .withColumn("categories", F.array())
)

print("🔵 A escrever índice (JSONL) em:", catalog_out)
(
    df_index
    .repartition(1)  # só para demo (mais simples de ler no backend)
    .write.mode("overwrite")
    .json(catalog_out)
)

spark.stop()
print("✅ books_index criado.")
