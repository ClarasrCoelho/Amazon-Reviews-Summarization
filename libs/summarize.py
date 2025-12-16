# libs/summarize.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from libs.nlp_udfs import textrank_top_sentences


_sum_tok = None
_sum_model = None


def _load_summarizer():
    global _sum_tok, _sum_model
    if _sum_tok is None or _sum_model is None:
        _sum_tok = AutoTokenizer.from_pretrained("t5-small")
        _sum_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").eval()
    return _sum_tok, _sum_model


def summarize_hybrid(
    text: str,
    num_sentences_textrank: int = 5,
    max_new_tokens: int = 128
) -> str:
    if not text:
        return ""

    # 1) extrativo (safe)
    base = textrank_top_sentences(
        text,
        top_k=num_sentences_textrank,
        max_chars=200_000
    )

    if not base:
        return ""

    # 2) abstrativo (T5)
    tok, model = _load_summarizer()

    enc = tok(
        "summarize: " + base,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )

    return tok.decode(out[0], skip_special_tokens=True)
