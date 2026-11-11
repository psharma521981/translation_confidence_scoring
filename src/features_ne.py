"""Number and named-entity consistency features.

No NER model is installed, so entities are found with a capitalisation heuristic
on the English source (capitalised words that are not sentence-initial, plus
all-caps acronyms). Hindi output transliterates most names into Devanagari, so
exact matching only detects entities copied in Latin script; the other
entities are counted but cannot be verified. Run from the project root:

    python src/features_ne.py
"""
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "features"

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Ordinals (2nd, 21st) are skipped: Hindi usually writes them as words
NUMBER = re.compile(r"\d+(?:[.,]\d+)*(?!\d|(?:st|nd|rd|th)\b)")
LATIN = re.compile(r"[A-Za-z]")
URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.]+")
EDGE = "".join(set(string.punctuation) | set("।॥“”‘’…–—«»"))
CONNECTORS = {"of", "the", "de", "van", "von", "for"}


def numbers(text):
    """Numbers as normalised strings: Devanagari digits -> ASCII, thousands commas dropped."""
    out = []
    for m in NUMBER.findall(text.translate(DEV_DIGITS)):
        m = m.rstrip(".,")
        # "1,000" / "1,00,000" -> thousands separators; "3,5" keeps its comma-as-decimal as-is
        if re.fullmatch(r"\d{1,3}(,\d{2,3})+(\.\d+)?", m):
            m = m.replace(",", "")
        out.append(m)
    return out


def entities(text):
    """Spans of capitalised tokens (not sentence-initial) and all-caps acronyms.

    A span ends at any punctuation, so "Jerry, Tom" gives two entities.
    """
    spans, cur = [], []
    acronyms = 0
    at_start = True  # first Latin token of the text or of a sentence
    for raw in text.split():
        t = re.sub(r"['’]s$", "", raw.strip(EDGE))
        if cur and raw[0] in EDGE:  # opening bracket/quote starts a new span: "Fund (IMF)"
            spans.append(cur)
            cur = []
        ends_clause = raw[-1] in EDGE
        if not t or not LATIN.match(t) or t == "I":
            if cur:
                spans.append(cur)
                cur = []
            at_start = at_start or ends_clause
            continue
        is_acr = len(t) >= 2 and t.isupper() and t.isalpha()
        is_cap = t[0].isupper() and not at_start
        if is_acr:
            acronyms += 1
        if is_acr or is_cap:
            cur.append(t)
        elif cur and t.lower() in CONNECTORS:
            cur.append(t)  # tentative; trimmed below if no capitalised word follows
            ends_clause = False
        else:
            if cur:
                spans.append(cur)
            cur = []
        at_start = raw[-1] in ".!?:]"
        if ends_clause and cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    out = []
    for s in spans:
        while s and s[-1].lower() in CONNECTORS:
            s = s[:-1]
        if s:
            out.append(" ".join(s))
    return out, acronyms



def latin_spans(text):
    """Runs of consecutive Latin-script tokens in the translation."""
    spans, cur = [], []
    for t in (t.strip(EDGE) for t in text.split()):
        if t and LATIN.search(t):
            cur.append(t.lower())
        elif cur:
            spans.append(" ".join(cur))
            cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans


def feature_row(src, mt):
    s_nums, m_nums = Counter(numbers(src)), Counter(numbers(mt))
    matched = sum((s_nums & m_nums).values())
    n_s, n_m = sum(s_nums.values()), sum(m_nums.values())
    missing = n_s - matched
    extra = n_m - matched

    ents, acronyms = entities(src)
    ent_tokens = sum(len(e.split()) for e in ents)
    mt_low = " " + " ".join(t.strip(EDGE).lower() for t in mt.split()) + " "
    copied = sum(f" {e.lower()} " in mt_low for e in ents)
    mt_spans = latin_spans(mt)
    ent_low = {e.lower() for e in ents}
    src_low_tokens = {t.strip(EDGE).lower() for t in src.split()}
    # Latin spans in the translation with no counterpart in the source
    foreign_spans = sum(not (set(s.split()) & src_low_tokens) for s in mt_spans)

    s_urls = URL.findall(src)
    n_words = max(len(src.split()), 1)

    return {
        # numbers
        "src_num_count": n_s, "mt_num_count": n_m,
        "num_count_diff": n_m - n_s,
        "has_src_numbers": int(n_s > 0),
        "nums_matched": matched, "nums_missing": missing, "nums_extra": extra,
        "num_match_frac": matched / n_s if n_s else 1.0,
        "all_numbers_match": int(missing == 0 and extra == 0),
        # named entities (heuristic)
        "src_entities": len(ents), "src_entity_tokens": ent_tokens,
        "src_acronyms": acronyms,
        "src_entity_density": ent_tokens / n_words,
        "has_src_entities": int(len(ents) > 0),
        "entities_copied_latin": copied,
        "entities_copied_frac": copied / len(ents) if ents else 0.0,
        "entities_not_copied": len(ents) - copied,
        "mt_latin_spans": len(mt_spans),
        "mt_foreign_latin_spans": foreign_spans,
        "mt_latin_spans_minus_entities": len(mt_spans) - len(ent_low),
        # URLs / emails should survive translation unchanged
        "src_urls": len(s_urls),
        "urls_lost": sum(u not in mt for u in s_urls),
    }


def main():
    FEATURES.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED / f"{split}.csv")
        feats = pd.DataFrame(
            [feature_row(s, m) for s, m in zip(df["original"], df["translation"])]
        )
        assert len(feats) == len(df) and not feats.isna().any().any()
        feats.to_csv(FEATURES / f"ne_{split}.csv", index=False)
        print(split, feats.shape)


if __name__ == "__main__":
    main()
