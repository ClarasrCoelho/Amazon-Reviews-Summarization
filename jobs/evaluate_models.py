# jobs/evaluate_models.py
import os
import sys
import yaml
from pyspark.sql import functions as F

from libs.spark_utils import build_spark

# garantir UTF-8 em Windows (evita cp1252 / UnicodeEncodeError)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    base_path = cfg["base_path"].rstrip("/")
    gold_root = cfg["paths"].get("gold", "gold")

    sentiment_path = f"{base_path}/{gold_root}/sentiment"
    eval_path = f"{base_path}/{gold_root}/eval"

    # garantir que Spark usa o python da venv
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = build_spark("labgdd-evaluate")
    spark.sparkContext.setLogLevel("WARN")

    print("A ler Gold/sentiment:", sentiment_path, flush=True)
    df = spark.read.parquet(sentiment_path)

    # precisamos de overall (rating) e sentiment_score
    if "overall" not in df.columns:
        raise RuntimeError("Gold/sentiment não tem 'overall'. Garante que veio do Silver.")
    if "sentiment_score" not in df.columns:
        raise RuntimeError("Gold/sentiment não tem 'sentiment_score'.")

    # label true: 1-2 neg, 4-5 pos, 3 ignora
    df2 = (
        df.withColumn(
            "y_true",
            F.when(F.col("overall").isin(1.0, 2.0, 1, 2), F.lit(0))
             .when(F.col("overall").isin(4.0, 5.0, 4, 5), F.lit(1))
             .otherwise(F.lit(None))
        )
        .filter(F.col("y_true").isNotNull())
    )

    # proxy prediction: sentiment_score >= 0 => pos
    df2 = df2.withColumn(
        "y_pred",
        F.when(F.col("sentiment_score") >= 0, F.lit(1)).otherwise(F.lit(0))
    )

    agg = df2.agg(
        F.sum(F.when((F.col("y_pred") == 1) & (F.col("y_true") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("y_pred") == 0) & (F.col("y_true") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("y_pred") == 1) & (F.col("y_true") == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("y_pred") == 0) & (F.col("y_true") == 1), 1).otherwise(0)).alias("fn"),
        F.count("*").alias("support"),
    )

    res = (
        agg
        .withColumn("precision", F.col("tp") / (F.col("tp") + F.col("fp")))
        .withColumn("recall",    F.col("tp") / (F.col("tp") + F.col("fn")))
        .withColumn("accuracy",  (F.col("tp") + F.col("tn")) / (F.col("tp") + F.col("tn") + F.col("fp") + F.col("fn")))
        .withColumn("f1",        2 * (F.col("precision") * F.col("recall")) / (F.col("precision") + F.col("recall")))
        .select("f1", "accuracy", "precision", "recall", "support")
    )

    print("A gravar Gold/eval:", eval_path, flush=True)
    res.coalesce(1).write.mode("overwrite").json(eval_path)

    spark.stop()
    print("Gold/eval concluído.", flush=True)


if __name__ == "__main__":
    main()
