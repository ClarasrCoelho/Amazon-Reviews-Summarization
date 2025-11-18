from langdetect import detect, LangDetectException

def detect_lang(text: str):
    if not text:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None
