import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import spacy
from libs.text_cleaning import split_sentences

# ---------- spaCy ----------
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

# ---------- Sentiment (BERT multi-lingue) ----------
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
    tok, model = _load_sentiment_model()
    texts_list = texts.fillna("").tolist()
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


# ---------- ABSA (aspectos simples por noun chunks) ----------
def extract_aspects_batch(texts: pd.Series) -> pd.Series:
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
    placeholder: devolve só lista de aspetos; a polaridade fina pode vir depois
    """
    return pd.DataFrame(
        {
            "aspects": aspects,
            "aspect_sentiments": aspects.apply(
                lambda x: ";".join(["unknown"] * len(x.split(";"))) if x else ""
            ),
        }
    )


# ---------- TextRank básico para frases-chave ----------
import networkx as nx

def textrank_top_sentences(text: str, num_sentences: int = 5) -> str:
    sents = split_sentences(text)
    if len(sents) <= num_sentences:
        return " ".join(sents)

    # grafo completo com peso simples por overlap de palavras
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
