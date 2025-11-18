from pyspark.sql import SparkSession
import yaml
import os

def load_config(path: str):
    """Lê um ficheiro YAML com as configs da pipeline."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_spark(app_name="ABGDD"):
    """Cria a SparkSession local."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

def load_raw(spark, path: str):
    """Lê dados raw em CSV, JSON ou Parquet."""
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".csv":
        return spark.read.option("header", True).csv(path)
    elif ext == ".json":
        return spark.read.json(path)
    elif ext == ".parquet":
        return spark.read.parquet(path)
    else:
        raise ValueError(f"Formato não suportado: {ext}")

def write_parquet(df, path: str, mode="overwrite"):
    df.write.mode(mode).parquet(path)
