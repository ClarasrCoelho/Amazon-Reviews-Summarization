import os
import sys
from pyspark.sql import SparkSession, functions as F
from libs.io_utils import load_config

# garantir que Spark usa o mesmo Python do driver (evita o "python3" inexistente no Windows)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

cfg = load_config("conf/pipeline.yaml")

BASE_PATH = cfg["base_path"].rstrip("/")
bronze_path = f"{BASE_PATH}/{cfg['paths']['bronze']}"
silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"

min_len = cfg.get("min_doc_length", 50)
target_lang = cfg.get("target_lang", "en")

spark = (
    SparkSession.builder
    .appName("labgdd-clean-normalize-s3")
    .master("local[*]")
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.profile.ProfileCredentialsProvider"
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")
    # força também nos executores (por segurança)
    .config("spark.executorEnv.PYSPARK_PYTHON", sys.executable)
    .getOrCreate()
)

print("🔵 A ler Bronze de:", bronze_path)
df = spark.read.parquet(bronze_path)

# usamos o text_norm já criado no ingest_batch
col_text = "text_norm" if "text_norm" in df.columns else "review_text"

print("🔵 A filtrar textos nulos...")
clean = df.filter(F.col(col_text).isNotNull())

print(f"🔵 A calcular comprimento de documento ({col_text})...")
clean = clean.withColumn("doc_len", F.length(F.col(col_text)))

print(f"🔵 A aplicar filtro de comprimento > {min_len}...")
clean = clean.filter(F.col("doc_len") > F.lit(min_len))

# filtro de língua se existir coluna lang e target_lang não for 'all'
if "lang" in clean.columns and target_lang and target_lang.lower() != "all":
    print(f"🔵 A filtrar pela língua = '{target_lang}'...")
    clean = clean.filter(F.col("lang") == target_lang)

print("🔵 Nº de registos após limpeza:", clean.count())

print("🔵 A gravar Silver em:", silver_path)
clean.write.mode("overwrite").parquet(silver_path)

spark.stop()
print("✅ Silver (S3) concluído.")
