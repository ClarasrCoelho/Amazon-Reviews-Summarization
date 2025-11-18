import re

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    # remover caracteres estranhos e normalizar espaços
    text = re.sub(r"[^\w\s.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def split_sentences(text: str):
    # split muito simples – podes trocar por nltk/sentencepiece depois
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
