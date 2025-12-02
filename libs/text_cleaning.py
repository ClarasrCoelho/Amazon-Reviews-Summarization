import re

def normalize_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # remover caracteres estranhos
    text = re.sub(r"[^\w\s.,!?-]", " ", text)
    # normalizar espaços
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def split_sentences(text):
    """
    Divide um texto em frases de forma simples, com base em . ! ?
    Devolve lista de strings.
    """
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)

    # separa em frases sempre que encontrar ., ! ou ? seguido de espaço
    parts = re.split(r'(?<=[.!?])\s+', text)
    # limpa espaços extra e remove vazios
    return [s.strip() for s in parts if s and s.strip()]
