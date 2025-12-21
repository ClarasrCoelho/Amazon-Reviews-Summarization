# libs/nlp_udfs.py
from __future__ import annotations

import re
from functools import lru_cache
from typing import List
import pandas as pd
import networkx as nx
import spacy

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType

# -------------------------
# spaCy (robusto)
# -------------------------
@lru_cache(maxsize=1)
def get_nlp():
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "textcat", "lemmatizer"])
    except Exception:
        nlp = spacy.blank("en")

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

def _simple_sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.strip()) > 20]

def textrank_top_sentences(text: str, top_k: int = 3) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        nlp = get_nlp()
        doc = nlp(text)
        try:
            sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
        except Exception:
            sentences = _simple_sentence_split(text)
    except Exception:
        sentences = _simple_sentence_split(text)

    if len(sentences) <= 1:
        return sentences[:top_k]

    def sent_to_words(s: str) -> set:
        return set(re.findall(r"[A-Za-z0-9]+", s.lower()))

    sent_words = [sent_to_words(s) for s in sentences]
    sim = [[0.0] * len(sentences) for _ in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            a, b = sent_words[i], sent_words[j]
            if not a or not b:
                continue
            sim[i][j] = len(a & b) / float(len(a | b))

    import numpy as np
    graph = nx.from_numpy_array(np.array(sim))
    scores = nx.pagerank(graph)
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return [s for _, s in ranked[:top_k]]

# -------------------------
# Lang detect (robusto)
# -------------------------
def detect_lang(text: str) -> str | None:
    try:
        from langdetect import detect
        from langdetect.lang_detect_exception import LangDetectException
    except Exception:
        return None

    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    try:
        return detect(t)
    except Exception:
        return None

# Spark UDFs (o que o injest_batch tenta importar)
lang_detect_udf = udf(detect_lang, StringType())

def split_sentences(text: str) -> List[str]:
    try:
        nlp = get_nlp()
        doc = nlp((text or "").strip())
        out = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 0]
        return out
    except Exception:
        return _simple_sentence_split(text or "")

sentence_split_udf = udf(split_sentences, ArrayType(StringType()))

# -------------------------
# Sentiment (batch)
# -------------------------
@lru_cache(maxsize=1)
def _get_transformers_pipe():
    try:
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True,
            max_length=256,
            local_files_only=True,
        )
    except Exception:
        return None

_POS_WORDS = {"good","great","excellent","amazing","love","loved","perfect","nice","awesome","best","wonderful","fantastic","recommend","recommended","happy","satisfied"}
_NEG_WORDS = {"bad","terrible","awful","hate","hated","poor","worst","broken","disappointed","waste","refund","return","returned","issue","problem","bugs","horrible"}

def _fallback_sentiment_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    denom = max(pos + neg, 1)
    score = (pos - neg) / float(denom)
    return float(max(-1.0, min(1.0, score)))

def predict_sentiment_batch(texts: pd.Series) -> pd.Series:
    pipe = _get_transformers_pipe()
    clean = texts.fillna("").astype(str)
    if pipe is None:
        return clean.apply(_fallback_sentiment_score)

    out_scores = []
    for t in clean.tolist():
        t = t.strip()
        if not t:
            out_scores.append(0.0)
            continue
        try:
            pred = pipe(t)[0]
            label = pred.get("label", "").upper()
            s = float(pred.get("score", 0.0))
            out_scores.append(s if label == "POSITIVE" else -s)
        except Exception:
            out_scores.append(_fallback_sentiment_score(t))
    return pd.Series(out_scores)

# -------------------------
# ABSA
# -------------------------
_POS_WORDS_ABSA = {"good","great","excellent","amazing","love","loved","nice","perfect","fast","smooth","easy","comfortable","worth","awesome","recommend"}
_NEG_WORDS_ABSA = {"bad","terrible","awful","poor","hate","hated","worse","worst","slow","broken","hard","difficult","uncomfortable","waste","disappointed","disappointing"}
_ASPECT_SEEDS = ["quality","price","value","delivery","shipping","packaging","design","size","fit","material","color","colour","battery","performance","speed","screen","sound","camera","comfort","durability","instructions","support"]

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())

def simple_aspect_polarity(text: str, aspect: str, window: int = 6) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    toks = _tokenize(text)
    if not toks:
        return 0.0
    aspect = (aspect or "").lower().strip()
    if not aspect:
        return 0.0
    idxs = [i for i, t in enumerate(toks) if t == aspect]
    if not idxs:
        return 0.0

    pos = 0
    neg = 0
    for i in idxs:
        lo = max(0, i - window)
        hi = min(len(toks), i + window + 1)
        ctx = toks[lo:hi]
        pos += sum(1 for w in ctx if w in _POS_WORDS_ABSA)
        neg += sum(1 for w in ctx if w in _NEG_WORDS_ABSA)

    if pos == 0 and neg == 0:
        return 0.0
    return float(pos - neg) / float(pos + neg)

def _extract_aspects_one(text: str, seeds: List[str] | None = None) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    seeds = seeds or _ASPECT_SEEDS
    low = text.lower()
    found = [a for a in seeds if re.search(rf"\b{re.escape(a)}\b", low)]
    if found:
        return found[:8]
    # fallback simples (sem POS tagging)
    toks = _tokenize(text)
    # apanha “candidatos” com >=4 chars e dedup
    out = []
    seen = set()
    for t in toks:
        if len(t) >= 4 and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= 8:
            break
    return out

def extract_aspects_batch(col: pd.Series) -> pd.Series:
    col = col.fillna("").astype(str)
    return col.apply(lambda t: _extract_aspects_one(t))
