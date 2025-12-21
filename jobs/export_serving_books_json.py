# jobs/export_serving_books_json.py
import json
import yaml
from urllib.parse import urlparse

import boto3
from libs.spark_utils import build_spark


def load_config(path: str = "conf/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_s3a_path(p: str):
    # s3a://bucket/path/to/dir
    if not p.startswith("s3a://"):
        raise ValueError(f"Esperava s3a://..., recebi: {p}")
    u = urlparse(p.replace("s3a://", "s3://", 1))
    bucket = u.netloc
    key = u.path.lstrip("/")
    return bucket, key


def main(asin: str | None = None):
    cfg = load_config()
    base = cfg["base_path"].rstrip("/")
    gold_root = cfg["paths"].get("gold", "gold")

    serving_parquet = f"{base}/{gold_root}/serving"              # parquet (do teu job)
    serving_books_prefix = f"{base}/{gold_root}/serving/books"   # onde vamos meter JSONs

    spark = build_spark("labgdd-export-serving-json")

    df = spark.read.parquet(serving_parquet)

    if asin:
        df = df.filter(df["asin"] == asin)

    rows = df.toJSON().collect()  # OK para 1 asin; para muitos, faz em batches

    spark.stop()

    # escrever no S3 com boto3 como objecto exacto: books/{asin}.json
    bucket, prefix = parse_s3a_path(serving_books_prefix)

    session = boto3.Session(profile_name=cfg.get("aws_profile", "default"))
    s3 = session.client("s3")

    wrote = 0
    for r in rows:
        obj = json.loads(r)
        a = obj.get("asin")
        if not a:
            continue
        key = f"{prefix}/{a}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
        )
        wrote += 1

    print(f"✅ Export JSON concluído. wrote={wrote} -> s3a://{bucket}/{prefix}/<asin>.json")


if __name__ == "__main__":
    import sys
    asin_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(asin_arg)
