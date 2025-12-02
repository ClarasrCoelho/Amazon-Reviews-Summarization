from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def detect_lang(text):
    """Deteta a língua do texto devolvendo código ISO ('en', 'pt', etc.)."""

    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    try:
        return detect(text)
    except LangDetectException:
        return None
    except Exception:
        return None
