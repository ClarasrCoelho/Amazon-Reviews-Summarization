import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import spacy
import langid
import networkx as nx

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType

from libs.text_cleaning import split_sentences

# ---------- spaCy (para aspetos) ----------
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        # modelo leve em inglês; ajusta se quiseres outro
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------- Sentiment (BERT multi-língue) ----------
_sent_tok = None
_sent_model = None

def _load_sentiment_model():
    global _sent_tok, _sent_model
    if _sent_tok is None or _sent_model is None:
        _sent_tok = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        _sent_model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        ).eval()
    return _sent_tok, _sent_model


def predict_sentiment_batch(texts: pd.Series) -> pd.Series:
    """
    Recebe uma Series de textos e devolve um score de 1–5 (média ponderada).
    Usado pelas pandas_udf de sentimento.
    """
    tok, model = _load_sentiment_model()
    texts_list = texts.fillna("").tolist()

    if not texts_list:
        return pd.Series(dtype=float)

    enc = tok(
        texts_list,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=256,
    )
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)
        scores = (probs * torch.arange(1, probs.shape[1] + 1)).sum(dim=1)

    return pd.Series(scores.cpu().numpy())


# ---------- ABSA (aspetos simples por noun chunks) ----------
def extract_aspects_batch(texts: pd.Series) -> pd.Series:
    """
    Extrai candidatos a aspetos (noun chunks 2-4 palavras) por texto
    e devolve uma string com ';' a separar.
    """
    nlp = get_nlp()
    aspects_list = []

    for txt in texts.fillna(""):
        doc = nlp(txt)
        cand = [
            nc.text.strip()
            for nc in doc.noun_chunks
            if 2 <= len(nc.text.split()) <= 4
        ]
        # únicos preservando ordem
        unique = list(dict.fromkeys(cand))
        aspects_list.append(";".join(unique))

    return pd.Series(aspects_list)


def simple_aspect_polarity(texts: pd.Series, aspects: pd.Series) -> pd.DataFrame:
    """
    Placeholder: devolve só a lista de aspetos;
    a polaridade fina fica com 'unknown' para cada aspeto.
    """
    return pd.DataFrame(
        {
            "aspects": aspects,
            "aspect_sentiments": aspects.apply(
                lambda x: ";".join(["unknown"] * len(x.split(";"))) if x else ""
            ),
        }
    )


# ---------- TextRank básico para frases-chave (usado no summarize) ----------
def textrank_top_sentences(text: str, num_sentences: int = 5) -> str:
    sents = split_sentences(text)
    if len(sents) <= num_sentences:
        return " ".join(sents)

    def sentence_similarity(s1, s2):
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / (len(w1 | w2) + 1e-9)

    g = nx.Graph()
    for i, s in enumerate(sents):
        g.add_node(i, sentence=s)

    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            sim = sentence_similarity(sents[i], sents[j])
            if sim > 0:
                g.add_edge(i, j, weight=sim)

    ranks = nx.pagerank(g, weight="weight")
    top_idx = sorted(ranks, key=ranks.get, reverse=True)[:num_sentences]
    top_idx = sorted(top_idx)
    return " ".join(sents[i] for i in top_idx)


# ---------- UDFs para Spark: língua e frases ----------
def _detect_lang(text: str) -> str:
    if text is None:
        return "unknown"
    lang, score = langid.classify(text)
    return lang

lang_detect_udf = udf(_detect_lang, StringType())


def _split_sentences_safe(text: str):
    if text is None:
        return []
    try:
        return split_sentences(text)
    except Exception:
        return []

sentence_split_udf = udf(_split_sentences_safe, ArrayType(StringType()))
