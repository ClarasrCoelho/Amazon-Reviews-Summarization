# libs/summarize.py
from __future__ import annotations

from typing import Optional

from libs.nlp_udfs import textrank_top_sentences

_sum_tok = None
_sum_model = None


def _try_load_t5() -> tuple[Optional[object], Optional[object]]:
    global _sum_tok, _sum_model
    if _sum_tok is not None and _sum_model is not None:
        return _sum_tok, _sum_model

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        _sum_tok = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
        _sum_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", local_files_only=True).eval()
        return _sum_tok, _sum_model
    except Exception:
        _sum_tok = None
        _sum_model = None
        return None, None


def summarize_hybrid(text: str, num_sentences_textrank: int = 5, max_new_tokens: int = 128) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # 1) Extrativo (TextRank)
    sents = textrank_top_sentences(text, top_k=num_sentences_textrank)
    base = " ".join(sents).strip()
    if not base:
        return ""

    # 2) Abstrativo (T5) se existir localmente
    tok, model = _try_load_t5()
    if tok is None or model is None:
        return base  # fallback: devolve extrativo

    import torch

    enc = tok("summarize: " + base, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)
