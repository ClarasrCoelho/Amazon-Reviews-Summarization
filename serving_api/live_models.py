import json
import time
from typing import Dict, Any, List

from libs.summarize import run_summary_models
from serving_api.s3_utils import read_text_from_s3, append_jsonl, write_text_to_s3

def _read_jsonl_list(bucket: str, key: str) -> List[dict]:
    try:
        txt = read_text_from_s3(bucket, key)
    except Exception:
        return []
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def recompute_live_for_asin(
    bucket: str,
    prefix: str,
    asin: str,
    base_summary: str,
) -> Dict[str, Any]:
    """
    Lê user_reviews do S3 -> corre 2 sumarizadores -> escolhe best -> grava resultados live no S3.
    """
    user_reviews_key = f"{prefix}/gold/user_reviews/{asin}.jsonl"
    user_reviews = _read_jsonl_list(bucket, user_reviews_key)

    # texto live = juntar user reviews (só o campo text)
    live_text = " ".join([(r.get("text") or "") for r in user_reviews]).strip()

    # modelos (comparação)
    models = run_summary_models(live_text if live_text else "")
    best = models["best"]

    # summary apresentado no livro:
    # - mantemos o base_summary (do dataset) e mostramos também o live_best
    # - se ainda não houver user reviews, live_best fica vazio
    live_payload = {
        "asin": asin,
        "ts": time.time(),
        "base_summary": base_summary or "",
        "live_reviews_count": len(user_reviews),
        "candidates": {
            "textrank_baseline": {
                "summary": models["textrank_baseline"].summary,
                "score": models["textrank_baseline"].score,
                "reasons": models["textrank_baseline"].reasons,
            },
            "abstractive_mapreduce": {
                "summary": models["abstractive_mapreduce"].summary,
                "score": models["abstractive_mapreduce"].score,
                "reasons": models["abstractive_mapreduce"].reasons,
            },
        },
        "best_model": best.name,
        "best_summary": best.summary,
        "best_score": best.score,
    }

    # gravar “live” por asin
    live_key = f"{prefix}/gold/live/books/{asin}.json"
    write_text_to_s3(bucket, live_key, json.dumps(live_payload, ensure_ascii=False), content_type="application/json")

    return live_payload
