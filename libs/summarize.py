# libs/summarize.py
from __future__ import annotations
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass

from libs.nlp_udfs import textrank_top_sentences

def _clean(text: str) -> str:
    t = (text or "")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(.)\1{6,}", r"\1\1\1", t)
    return t

def summarize_baseline_textrank(text: str, top_k: int = 5) -> str:
    text = _clean(text)
    if not text:
        return ""
    sents = textrank_top_sentences(text, top_k=top_k)
    return " ".join(sents).strip()

# -------- Abstrativo (opcional, só se existir LOCAL) --------
_ABS_MODEL_NAME = "google/flan-t5-base"

def _abs_load():
    """
    Só carrega se existir em cache/local (sem internet).
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(_ABS_MODEL_NAME, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(_ABS_MODEL_NAME, local_files_only=True).eval()
        return tok, model
    except Exception:
        return None, None

def _chunk_by_chars(text: str, max_chars: int = 4000) -> List[str]:
    """
    Chunk simples por chars (mais previsível no Windows).
    """
    text = _clean(text)
    if not text:
        return []
    parts = []
    for i in range(0, len(text), max_chars):
        parts.append(text[i:i+max_chars])
        if len(parts) >= 20:
            break
    return parts

def _abs_generate(tok, model, prompt: str, max_new_tokens: int = 120) -> str:
    import torch
    # T5 costuma ter max input ~512 tokens -> truncation True resolve
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()

def summarize_abstractive_mapreduce(text: str, max_new_tokens: int = 128) -> str:
    text = _clean(text)
    if not text:
        return ""

    tok, model = _abs_load()
    if tok is None or model is None:
        # offline -> baseline
        return summarize_baseline_textrank(text, top_k=5)

    chunks = _chunk_by_chars(text, max_chars=4000)
    if not chunks:
        return ""

    partials = []
    for c in chunks:
        prompt = (
            "Write a concise summary of the following book reviews. "
            "Avoid copying sentences verbatim; merge ideas and remove repetition:\n\n"
            f"{c}"
        )
        partials.append(_abs_generate(tok, model, prompt, max_new_tokens=120))

    merged = " ".join([p for p in partials if p]).strip()
    if not merged:
        return ""

    final_prompt = (
        "Combine the following partial summaries into ONE coherent final summary. "
        "Keep it clear and non-repetitive:\n\n"
        f"{merged}"
    )
    return _abs_generate(tok, model, final_prompt, max_new_tokens=max_new_tokens)

def summarize_hybrid(text: str, num_sentences_textrank: int = 5, max_new_tokens: int = 128) -> str:
    # mantém assinatura
    return summarize_abstractive_mapreduce(text, max_new_tokens=max_new_tokens)

# ---------------------------------------------------------
# Compat layer: run_summary_models() (usado pelo serving)
# ---------------------------------------------------------

@dataclass
class ModelSummary:
    name: str
    summary: str
    score: float
    reasons: List[str]

def _score_summary(summary: str, source_text: str) -> Tuple[float, List[str]]:
    """
    Score simples e determinístico (sem dependências externas):
    - penaliza vazio
    - prefere resumos com tamanho razoável
    - penaliza repetição (bigrams repetidos)
    - bónus leve por "compressão" (resumo bem menor que fonte)
    """
    summary = _clean(summary)
    src = _clean(source_text)

    if not summary:
        return 0.0, ["empty"]

    sum_len = len(summary)
    src_len = max(len(src), 1)

    # alvo ~ 400..1200 chars (ajusta como quiseres)
    if sum_len < 150:
        len_score = 0.4
        len_reason = "too_short"
    elif sum_len > 2500:
        len_score = 0.6
        len_reason = "too_long"
    else:
        len_score = 1.0
        len_reason = "ok_length"

    # repetição por bigramas
    toks = re.findall(r"[A-Za-zÀ-ÿ0-9']+", summary.lower())
    bigrams = list(zip(toks, toks[1:])) if len(toks) >= 2 else []
    rep_pen = 0.0
    if bigrams:
        from collections import Counter
        c = Counter(bigrams)
        rep_ratio = 1.0 - (len(c) / max(len(bigrams), 1))
        rep_pen = min(0.4, rep_ratio)  # cap
    # compressão
    comp = sum_len / src_len
    comp_bonus = 0.15 if comp < 0.35 else 0.0

    score = (1.0 * len_score) + comp_bonus - rep_pen
    score = max(0.0, min(1.0, score))
    reasons = [len_reason]
    if comp_bonus > 0:
        reasons.append("good_compression")
    if rep_pen > 0.1:
        reasons.append("repetitive")

    return score, reasons

def run_summary_models(text: str, *, top_k: int = 5, max_new_tokens: int = 128) -> Dict[str, ModelSummary]:
    """
    API esperada pelo teu serving:
      out = run_summary_models(concat)
      b = out["best"]
      b.name / b.summary / b.score / b.reasons
    """
    src = _clean(text)

    s1 = summarize_baseline_textrank(src, top_k=top_k)
    sc1, rs1 = _score_summary(s1, src)

    s2 = summarize_abstractive_mapreduce(src, max_new_tokens=max_new_tokens)
    sc2, rs2 = _score_summary(s2, src)

    m1 = ModelSummary("textrank_baseline", s1, float(sc1), rs1)
    m2 = ModelSummary("abstractive_mapreduce", s2, float(sc2), rs2)

    best = m1 if m1.score >= m2.score else m2

    return {
        "textrank_baseline": m1,
        "abstractive_mapreduce": m2,
        "best": best,
    }
