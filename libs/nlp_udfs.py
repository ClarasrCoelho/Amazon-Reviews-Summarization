import langid
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType
from libs.text_cleaning import split_sentences

def _detect_lang(text: str) -> str:
    if text is None:
        return "unknown"
    lang, score = langid.classify(text)
    return lang

lang_detect_udf = udf(_detect_lang, StringType())

def _split_sentences(text: str):
    if text is None:
        return []
    try:
        return split_sentences(text)
    except:
        return []

sentence_split_udf = udf(_split_sentences, ArrayType(StringType()))
