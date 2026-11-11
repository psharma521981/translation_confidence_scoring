"""Preprocess the raw English-Hindi QE data.

Steps: pool train+dev, normalize Unicode, drop empty rows, drop exact repeats of
(source, translation) keeping the first, then split by source sentence into a
train+val file and a fixed test file. Run from the project root:

    python src/preprocess.py
"""
import json
import unicodedata
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_raw():
    frames = []
    for name in ("train", "dev"):
        # quoting=3 (QUOTE_NONE): the text contains quote characters
        df = pd.read_csv(RAW / f"{name}.enhi.df.short.tsv", sep="\t", quoting=3)
        df["origin"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def normalize(text):
    return unicodedata.normalize("NFC", str(text)).strip()


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    seed, test_size = cfg["seed"], cfg["test_size"]

    df = load_raw()
    stats = {"rows_raw": len(df), "rows_raw_train": int((df.origin == "train").sum()),
             "rows_raw_dev": int((df.origin == "dev").sum())}

    df["original"] = df["original"].map(normalize)
    df["translation"] = df["translation"].map(normalize)

    empty = (df["original"] == "") | (df["translation"] == "")
    stats["rows_empty_removed"] = int(empty.sum())
    df = df[~empty]

    before = len(df)
    df = df.drop_duplicates(["original", "translation"], keep="first")
    stats["rows_duplicates_removed"] = before - len(df)
    stats["rows_clean"] = len(df)
    stats["unique_sources"] = int(df["original"].nunique())
    df = df.reset_index(drop=True)

    # Split by source sentence so one source never appears in two splits.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(splitter.split(df, groups=df["original"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]
    assert not set(trainval["original"]) & set(test["original"]), "source leakage"

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    trainval.to_csv(OUT / "trainval.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    df.to_csv(OUT / "clean_all.csv", index=False)

    stats.update({"rows_trainval": len(trainval), "rows_test": len(test),
                  "seed": seed, "test_size": test_size})
    (RESULTS / "preprocess_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
