import os
import sys
import yaml
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StringType, DoubleType

from libs.nlp_udfs import extract_aspects_batch, simple_aspect_polarity


# -------------------------
# Windows / Hadoop winutils
# -------------------------
HADOOP_HOME = os.getenv("HADOOP_HOME", r"C:\hadoop")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME
os.environ["PATH"] = os.path.join(HADOOP_HOME, "bin") + ";" + os.environ.get("PATH", "")

# Garantir que Spark usa o python da venv
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
BASE_PATH = cfg["base_path"].rstrip("/")
silver_path = f"{BASE_PATH}/{cfg['paths']['silver']}"
gold_root = cfg["paths"].get("gold", "gold")
absa_path = f"{BASE_PATH}/{gold_root}/absa"

# Dirs temporários (Windows)
if os.name == "nt":
    os.makedirs(r"C:\spark-tmp", exist_ok=True)
    os.makedirs(r"C:\spark-warehouse", exist_ok=True)

spark = (
    SparkSession.builder
    .appName("labgdd-04-absa-aspects")
    .master("local[*]")

    # Hadoop home dir para winutils
    .config("spark.driver.extraJavaOptions", f'-Dhadoop.home.dir="{HADOOP_HOME}"')
    .config("spark.executor.extraJavaOptions", f'-Dhadoop.home.dir="{HADOOP_HOME}"')

    # temp/warehouse (evita stresses Windows)
    .config("spark.local.dir", "C:/spark-tmp")
    .config("spark.sql.warehouse.dir", "C:/spark-warehouse")

    # S3A
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.profile.ProfileCredentialsProvider"
    )
    .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com")

    # Arrow + timeouts
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
    .config("spark.network.timeout", "600s")
    .config("spark.executor.heartbeatInterval", "60s")

    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("🔵 A ler Silver de:", silver_path)
df = spark.read.parquet(silver_path)

text_col = "text_norm" if "text_norm" in df.columns else ("review_text" if "review_text" in df.columns else None)
if text_col is None:
    raise RuntimeError("Não encontro coluna de texto (text_norm ou review_text) no Silver.")

df = df.repartition(4)


# -------------------------
# UDFs
# -------------------------
@pandas_udf(ArrayType(StringType()))
def aspects_udf(texts: pd.Series) -> pd.Series:
    # garantir nunca None e sempre lista por linha
    res = extract_aspects_batch(texts)
    if res is None:
        return pd.Series([[]] * len(texts))

    out = []
    for r in res:
        if r is None:
            out.append([])
        elif isinstance(r, np.ndarray):
            out.append(r.tolist())
        elif isinstance(r, (list, tuple)):
            out.append(list(r))
        else:
            try:
                out.append(list(r))
            except Exception:
                out.append([])
    return pd.Series(out)


@pandas_udf(DoubleType())
def avg_aspect_polarity_udf(texts: pd.Series, aspects: pd.Series) -> pd.Series:
    texts = texts.fillna("").astype(str)

    def normalize_aspects(a):
        # a pode vir: None, list, numpy.ndarray, ou float (NaN)
        if a is None:
            return []
        if isinstance(a, float) and pd.isna(a):
            return []
        if isinstance(a, np.ndarray):
            return a.tolist()
        if isinstance(a, (list, tuple)):
            return list(a)
        try:
            return list(a)
        except Exception:
            return []

    def row_score(t: str, a):
        asp_list = normalize_aspects(a)
        if len(asp_list) == 0:
            return 0.0
        scores = [simple_aspect_polarity(t, asp) for asp in asp_list if asp]
        return float(sum(scores)) / float(len(scores)) if scores else 0.0

    out = [row_score(t, a) for t, a in zip(texts.tolist(), aspects.tolist())]
    return pd.Series(out, dtype="float64")


print(f"🔵 A extrair aspetos a partir de '{text_col}'...")
df2 = df.withColumn("aspects", aspects_udf(F.col(text_col)))

print("🔵 A calcular polaridade média por aspeto...")
df2 = df2.withColumn("avg_aspect_polarity", avg_aspect_polarity_udf(F.col(text_col), F.col("aspects")))

select_cols = []
if "review_id" in df2.columns:
    select_cols.append("review_id")
if "reviewerID" in df2.columns:
    select_cols.append("reviewerID")
if "asin" in df2.columns:
    select_cols.append("asin")

df_absa = df2.select(*select_cols, "aspects", "avg_aspect_polarity")

print("🔵 A gravar ABSA em:", absa_path)
df_absa.write.mode("overwrite").parquet(absa_path)

spark.stop()
print("✅ Gold/ABSA no S3 concluído.")
