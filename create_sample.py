import itertools

in_path = r"data/raw/Books.json"
out_path = r"data/raw/Books_sample.json"

max_lines = 500_000  # ajusta: 100_000, 1_000_000, etc.

count = 0
with open(in_path, "r", encoding="utf-8") as fin, \
     open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(line)
        count += 1
        if count >= max_lines:
            break

print(f"Escrevi {count} linhas para {out_path}")
