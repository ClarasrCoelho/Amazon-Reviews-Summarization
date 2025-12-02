import os
import glob
import pandas as pd
import langid
from libs.io_utils import load_config
from libs.text_cleaning import normalize_text

# limitar aos idiomas que te interessam
langid.set_languages(['en', 'pt'])

# carregar config
cfg = load_config("conf/pipeline.yaml")
BASE_PATH = cfg["base_path"]
silver_dir = os.path.join(BASE_PATH, cfg["paths"]["silver"])   # ex: data/silver/reviews
out_path = os.path.join(BASE_PATH, "silver", "reviews_with_lang.parquet")

print("SILVER_DIR:", silver_dir)

def detect_lang_offline(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    code, _ = langid.classify(text)
    return code

# 1) ler todos os ficheiros parquet do silver (produzidos pelo Spark)
files = glob.glob(os.path.join(silver_dir, "*.parquet"))
if not files:
    raise RuntimeError(f"Nenhum ficheiro parquet encontrado em {silver_dir}")

print(f"A ler {len(files)} ficheiros de Silver...")
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print("Registos lidos:", len(df))

# 2) garantir coluna de texto normalizado (se quiseres refazer)
if "review_text_norm" not in df.columns:
    print("Coluna review_text_norm não existe, a criar com normalize_text()...")
    df["review_text_norm"] = df["review_text"].apply(normalize_text)

# 3) detetar língua com base no texto normalizado
print("A detetar língua com langid...")
df["lang"] = df["review_text_norm"].apply(detect_lang_offline)

# 4) gravar novo parquet com língua
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_parquet(out_path, index=False)

print("✅ Ficheiro com língua gravado em:", out_path)
