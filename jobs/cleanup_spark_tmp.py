# jobs/cleanup_spark_tmp.py
from __future__ import annotations
import os
import shutil
from pathlib import Path

def main() -> None:
    root = Path(os.environ.get("SPARK_TMP_ROOT", r"C:\spark-tmp"))
    if not root.exists():
        print("✅ spark tmp root não existe:", root)
        return

    removed = 0
    failed = 0

    for p in root.glob("run-*"):
        # apaga runs antigos (não o do PID actual)
        if f"run-{os.getpid()}" in p.name:
            continue
        try:
            shutil.rmtree(p, ignore_errors=False)
            removed += 1
        except Exception:
            # no Windows pode falhar se ainda houver lock; ignora
            failed += 1

    print(f"✅ cleanup: removed={removed} failed={failed} root={root}")

if __name__ == "__main__":
    main()
