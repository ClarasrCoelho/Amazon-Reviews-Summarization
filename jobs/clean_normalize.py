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

min_len = int(cfg.get("min_doc_length", 50))
target_lang = cfg.get("target_lang", "en")

# Ajusta conforme a tua máquina (4/8 costuma ser bom em Windows + S3)
NUM_PARTS = int(os.getenv("LABGDD_PARTS", "8"))

spark = (
    SparkSession.builder
    .appName("labgdd-clean-normalize-s3")
    .master("local[*]")

    # Hadoop AWS (S3A)
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")

    # Credenciais via profile (~/.aws/credentials). (Mantive a tua opção)
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.profile.ProfileCredentialsProvider"
    )

    # Região/endpoint
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")
    .config("spark.hadoop.fs.s3a.path.style.access", "false")

    # --- Estabilidade S3A (timeouts / retries) ---
    .config("spark.hadoop.fs.s3a.connection.maximum", "64")
    .config("spark.hadoop.fs.s3a.threads.max", "64")
    .config("spark.hadoop.fs.s3a.attempts.maximum", "10")
    .config("spark.hadoop.fs.s3a.retry.limit", "10")
    .config("spark.hadoop.fs.s3a.retry.throttle.limit", "10")
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
    .config("spark.hadoop.fs.s3a.socket.timeout", "120000")

    # Upload mais robusto (multipart)
    .config("spark.hadoop.fs.s3a.fast.upload", "true")
    .config("spark.hadoop.fs.s3a.multipart.size", "67108864")   # 64MB
    .config("spark.hadoop.fs.s3a.multipart.threshold", "67108864")

    # Evita cache estranha do FS em jobs sequenciais no Windows
    .config("spark.hadoop.fs.s3a.impl.disable.cache", "true")

    # --- Commit/overwrite em S3 (menos dores) ---
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    .config("spark.hadoop.fs.s3a.committer.name", "directory")
    .config("spark.hadoop.fs.s3a.committer.staging.conflict-mode", "replace")

    # --- Menos paralelismo para não matar a heap + S3 ---
    .config("spark.sql.shuffle.partitions", str(NUM_PARTS))
    .config("spark.default.parallelism", str(NUM_PARTS))

    # força também nos executores (por segurança)
    .config("spark.executorEnv.PYSPARK_PYTHON", sys.executable)

    .getOrCreate()
)

try:
    print("🔵 A ler Bronze de:", bronze_path)

    # Se suspeitares de ficheiros parquet incompletos no bronze, ativa isto:
    # spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")
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

    # IMPORTANTE: reduzir nº de ficheiros/writers antes de escrever no S3
    clean = clean.coalesce(NUM_PARTS)

    # Evita um count() gigante antes do write (se quiseres mesmo, deixa)
    # print("🔵 Nº de registos após limpeza:", clean.count())
    print("🔵 A gravar Silver em:", silver_path)

    clean.write.mode("overwrite").parquet(silver_path)

    print("✅ Silver (S3) concluído.")
finally:
    spark.stop()
