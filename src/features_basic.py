"""Length, punctuation, script and overlap features (no model needed).

Features are deterministic functions of (source, translation), so they are
computed for train, val and test with the same code. Output rows keep the order
of the input CSVs. Run from the project root:

    python src/features_basic.py
"""
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "features"

PUNCT = set(string.punctuation) | set("।॥“”‘’…–—«»")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9०-९]")  # ASCII and Devanagari digits


def tokens(text):
    # whitespace split + edge punctuation strip; \w would split Hindi at vowel signs
    toks = (t.strip("".join(PUNCT)) for t in text.split())
    return [t for t in toks if t]


def n_punct(text):
    return sum(ch in PUNCT for ch in text)


def feature_row(src, mt):
    s_tok, m_tok = tokens(src), tokens(mt)
    s_low = [t.lower() for t in s_tok]
    m_low = [t.lower() for t in m_tok]
    s_n, m_n = max(len(s_tok), 1), max(len(m_tok), 1)
    s_chars, m_chars = max(len(src), 1), max(len(mt), 1)

    # Cross-lingual overlap: tokens that appear unchanged in both sides. For
    # English->Hindi these are mostly numbers, names and untranslated words.
    shared = set(s_low) & set(m_low)
    latin_in_mt = [t for t in m_low if LATIN.search(t)]
    copied_latin = [t for t in latin_in_mt if t in set(s_low)]

    m_counts = Counter(m_low)
    repeats = sum(c - 1 for c in m_counts.values())
    bigrams = list(zip(m_low, m_low[1:]))
    rep_bigrams = len(bigrams) - len(set(bigrams))

    return {
        # length
        "src_chars": len(src), "mt_chars": len(mt),
        "src_words": len(s_tok), "mt_words": len(m_tok),
        "word_ratio": len(m_tok) / s_n,
        "char_ratio": len(mt) / s_chars,
        "word_diff": len(m_tok) - len(s_tok),
        "src_avg_word_len": np.mean([len(t) for t in s_tok]) if s_tok else 0.0,
        "mt_avg_word_len": np.mean([len(t) for t in m_tok]) if m_tok else 0.0,
        # punctuation
        "src_punct": n_punct(src), "mt_punct": n_punct(mt),
        "punct_diff": n_punct(mt) - n_punct(src),
        "mt_ends_danda_or_period": int(mt.rstrip()[-1:] in ("।", ".", "?", "!")),
        "src_ends_terminal": int(src.rstrip()[-1:] in (".", "?", "!")),
        "src_commas": src.count(","), "mt_commas": mt.count(","),
        "src_quotes": sum(src.count(q) for q in "\"'“”‘’"),
        "mt_quotes": sum(mt.count(q) for q in "\"'“”‘’"),
        "src_brackets": sum(src.count(b) for b in "()[]"),
        "mt_brackets": sum(mt.count(b) for b in "()[]"),
        # script mix in the translation
        "mt_devanagari_frac": len(DEVANAGARI.findall(mt)) / m_chars,
        "mt_latin_frac": len(LATIN.findall(mt)) / m_chars,
        "mt_latin_words": len(latin_in_mt),
        "mt_latin_word_frac": len(latin_in_mt) / m_n,
        "src_digits": len(DIGIT.findall(src)), "mt_digits": len(DIGIT.findall(mt)),
        # overlap
        "shared_tokens": len(shared),
        "shared_frac_src": len(shared) / max(len(set(s_low)), 1),
        "shared_frac_mt": len(shared) / max(len(set(m_low)), 1),
        "copied_latin_words": len(copied_latin),
        "copied_latin_frac_mt": len(copied_latin) / m_n,
        "src_uppercase_words": sum(t[:1].isupper() for t in s_tok[1:]),
        # fluency proxies: repetition and lexical diversity in the translation
        "mt_type_token_ratio": len(m_counts) / m_n,
        "mt_repeated_tokens": repeats,
        "mt_repeated_bigrams": rep_bigrams,
        "src_type_token_ratio": len(set(s_low)) / s_n,
    }


def main():
    FEATURES.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED / f"{split}.csv")
        feats = pd.DataFrame(
            [feature_row(s, m) for s, m in zip(df["original"], df["translation"])]
        )
        assert len(feats) == len(df) and not feats.isna().any().any()
        feats.to_csv(FEATURES / f"basic_{split}.csv", index=False)
        print(split, feats.shape)


if __name__ == "__main__":
    main()
