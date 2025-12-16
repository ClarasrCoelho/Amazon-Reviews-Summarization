# C:\LABGDD\libs\nlp_udfs.py
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Tuple

import networkx as nx
import spacy
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=1)
def get_nlp():
    """
    Carrega spaCy de forma robusta:
    - tenta en_core_web_sm (se existir)
    - caso não exista, usa blank('en')
    - garante SEMPRE sentence boundaries via sentencizer
    - desativa coisas pesadas para performance
    """
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "textcat", "lemmatizer"])
    except Exception:
        nlp = spacy.blank("en")

    # Garantir sentence boundaries
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    return nlp


def _simple_sentence_split(text: str) -> List[str]:
    """
    Fallback se mesmo assim alguma coisa falhar.
    Split simples por pontuação forte.
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.strip()) > 20]


def textrank_top_sentences(text: str, top_k: int = 3) -> List[str]:
    """
    Devolve top_k frases por TextRank.
    Usa spaCy para segmentação, mas tem fallback para split simples.
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        nlp = get_nlp()
        doc = nlp(text)

        # Se por alguma razão doc.sents falhar, apanha e faz fallback
        try:
            sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
        except Exception:
            sentences = _simple_sentence_split(text)

    except Exception:
        sentences = _simple_sentence_split(text)

    if len(sentences) <= 1:
        return sentences[:top_k]

    # Similaridade simples entre frases (bag-of-words)
    def sent_to_words(s: str) -> set:
        return set(re.findall(r"[A-Za-z0-9]+", s.lower()))

    sent_words = [sent_to_words(s) for s in sentences]

    # Matriz de similaridade
    sim = [[0.0] * len(sentences) for _ in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            a, b = sent_words[i], sent_words[j]
            if not a or not b:
                continue
            sim[i][j] = len(a & b) / float(len(a | b))

    graph = nx.from_numpy_array(__import__("numpy").array(sim))
    scores = nx.pagerank(graph)

    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return [s for _, s in ranked[:top_k]]

# ---------- Sentiment (batch) ----------

@lru_cache(maxsize=1)
def _get_transformers_pipe():
    """
    Tenta carregar um pipeline de sentiment do Transformers.
    IMPORTANTE: usa local_files_only=True para não tentar ir à internet.
    Se não houver modelo em cache/local, devolve None e usamos fallback.
    """
    try:
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True,
            max_length=256,
            local_files_only=True,  # <- não faz download
        )
    except Exception:
        return None


_POS_WORDS = {
    "good","great","excellent","amazing","love","loved","perfect","nice","awesome",
    "best","wonderful","fantastic","recommend","recommended","happy","satisfied"
}
_NEG_WORDS = {
    "bad","terrible","awful","hate","hated","poor","worst","broken","disappointed",
    "waste","refund","return","returned","issue","problem","bugs","horrible"
}

def _fallback_sentiment_score(text: str) -> float:
    """
    Fallback offline: score simples baseado em léxico.
    Devolve float em [-1, 1].
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0

    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    denom = max(pos + neg, 1)
    score = (pos - neg) / float(denom)

    # limitar a [-1, 1]
    if score > 1.0: score = 1.0
    if score < -1.0: score = -1.0
    return float(score)


def predict_sentiment_batch(texts: pd.Series) -> pd.Series:
    """
    Entrada: pd.Series de strings
    Saída: pd.Series float (sentiment_score) em [-1, 1]
    - Tenta Transformers se houver modelo em cache/local.
    - Caso contrário usa fallback lexicon (offline).
    """
    pipe = _get_transformers_pipe()

    # normalizar
    clean = texts.fillna("").astype(str)

    if pipe is None:
        return clean.apply(_fallback_sentiment_score)

    # Transformers (SST-2): POSITIVE/NEGATIVE com score [0..1]
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

# --- ABSA helpers (aspects + polarity) ---------------------------------------

# Lista simples de termos positivos/negativos (podes ajustar)
_POS_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "nice", "perfect",
    "fast", "smooth", "easy", "comfortable", "worth", "awesome", "recommend"
}
_NEG_WORDS = {
    "bad", "terrible", "awful", "poor", "hate", "hated", "worse", "worst",
    "slow", "broken", "hard", "difficult", "uncomfortable", "waste",
    "disappointed", "disappointing"
}

# Aspects “seed” típicos para Amazon reviews (podes alterar conforme o teu dataset)
_ASPECT_SEEDS = [
    "quality", "price", "value", "delivery", "shipping", "packaging",
    "design", "size", "fit", "material", "color", "colour",
    "battery", "performance", "speed", "screen", "sound", "camera",
    "comfort", "durability", "instructions", "support"
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def simple_aspect_polarity(text: str, aspect: str, window: int = 6) -> float:
    """
    Polaridade simples em torno de um 'aspect' (seed):
    - procura ocorrências do aspect no texto
    - conta palavras positivas/negativas numa janela de tokens
    - devolve score entre [-1, 1]
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    toks = _tokenize(text)
    if not toks:
        return 0.0

    aspect = (aspect or "").lower().strip()
    if not aspect:
        return 0.0

    # encontrar posições do aspect (token exacto)
    idxs = [i for i, t in enumerate(toks) if t == aspect]
    if not idxs:
        return 0.0

    pos = 0
    neg = 0
    for i in idxs:
        lo = max(0, i - window)
        hi = min(len(toks), i + window + 1)
        ctx = toks[lo:hi]
        pos += sum(1 for w in ctx if w in _POS_WORDS)
        neg += sum(1 for w in ctx if w in _NEG_WORDS)

    if pos == 0 and neg == 0:
        return 0.0
    return float(pos - neg) / float(pos + neg)


def _extract_aspects_one(text: str, seeds: List[str] = None) -> List[str]:
    """
    Extrai aspects de forma simples:
    - se encontrar seeds no texto, devolve essas
    - caso contrário tenta apanhar 'noun-like words' pelo spaCy (se existir)
    """
    if not isinstance(text, str) or not text.strip():
        return []

    seeds = seeds or _ASPECT_SEEDS
    low = text.lower()

    found = [a for a in seeds if re.search(rf"\b{re.escape(a)}\b", low)]
    if found:
        return found[:8]  # limitar

    # fallback: usar spaCy para apanhar substantivos (se tiver tagger/parser não existir, isto falha -> apanha except)
    try:
        nlp = get_nlp()
        doc = nlp(text)
        # se o pipeline for blank+sentencizer, não há POS -> isto pode dar vazio
        nouns = []
        for tok in doc:
            if getattr(tok, "pos_", "") in ("NOUN", "PROPN"):
                t = tok.text.lower().strip()
                if len(t) >= 3 and t.isalpha():
                    nouns.append(t)
        # dedup preservando ordem
        seen = set()
        out = []
        for n in nouns:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out[:8]
    except Exception:
        return []


def extract_aspects_batch(col: pd.Series) -> pd.Series:
    """
    Pandas batch: devolve lista de aspects por linha.
    Compatível com pandas_udf(ArrayType(StringType()))
    """
    col = col.fillna("").astype(str)
    return col.apply(lambda t: _extract_aspects_one(t))


def aspect_polarity_batch(texts: pd.Series, aspects: pd.Series) -> pd.Series:
    """
    Opcional: calcula polaridade média por linha, dado texto + lista de aspects.
    """
    texts = texts.fillna("").astype(str)

    def _row_score(t: str, asp_list) -> float:
        if not asp_list:
            return 0.0
        scores = [simple_aspect_polarity(t, a) for a in asp_list]
        if not scores:
            return 0.0
        return float(sum(scores)) / float(len(scores))

    return pd.Series([_row_score(t, a) for t, a in zip(texts, aspects)])
