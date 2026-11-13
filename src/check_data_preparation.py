"""Satu cek runnable: verifikasi artefak hasil 02_data_preparation.ipynb."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

train, val, test = (pd.read_csv(P / f"{s}.csv") for s in ("train", "val", "test"))
manifest = json.loads((P / "split_manifest.json").read_text(encoding="utf-8"))

# 1. Jumlah baris sesuai manifest
for name, df in (("train", train), ("val", val), ("test", test)):
    assert len(df) == manifest["split"][name], (name, len(df))

# 2. Tidak ada kebocoran teks antar-split
assert not (set(train["Tweet"]) & set(test["Tweet"]))
assert not (set(train["Tweet"]) & set(val["Tweet"]))
assert not (set(val["Tweet"]) & set(test["Tweet"]))

# 3. Test set belum berubah sejak dibuat (hash di manifest)
assert hashlib.sha256("\n".join(test["Tweet"]).encode("utf-8")).hexdigest() == \
    manifest["sha256_teks_test"]

# 4. Tidak ada dokumen kosong & tidak ada kolom konteks yang hilang
for df in (train, val, test):
    assert df["text_clean_contextual"].str.strip().ne("").all()
    assert {"Abusive", "HS_Individual", "is_gov"}.issubset(df.columns)

# 5. Proporsi kelas tetap sama di semua split (±2 poin persentase)
rates = [df["HS"].mean() for df in (train, val, test)]
assert max(rates) - min(rates) < 0.02, rates

print("OK semua cek lolos:", manifest["split"],
      "| proporsi HS:", [round(r, 4) for r in rates])
