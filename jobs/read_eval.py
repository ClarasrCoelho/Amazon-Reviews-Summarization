# jobs/read_eval.py
from __future__ import annotations

import yaml
from libs.spark_utils import build_spark


def load_config(path: str = "conf/pipeline.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    base_path = str(cfg["base_path"]).rstrip("/")
    gold_root = str(cfg.get("paths", {}).get("gold", "gold")).strip("/")
    eval_path = f"{base_path}/{gold_root}/eval"

    spark = build_spark("labgdd-read-eval")

    print("🔵 A ler Gold/eval:", eval_path)
    df = spark.read.json(eval_path)
    df.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
