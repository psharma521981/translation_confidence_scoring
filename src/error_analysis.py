"""Error analysis by sentence type (RQ4).

Which kinds of sentence pairs fail most, and where does the blend model go wrong?
Uses the validation and test rows pooled (1,643 rows) for more statistical power.
This is descriptive only: no model, config or threshold is changed here. The blend
threshold was tuned on validation, so recall on the validation part is slightly
optimistic. Outputs go to results/error_analysis/. Run from the project root:

    python src/error_analysis.py
"""
import re
import string
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import load_cutoff

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "features"
FINAL = ROOT / "results" / "final"
OUT = ROOT / "results" / "error_analysis"
BLEND = "blend: xgboost (all) + xlmr"
EDGE = "".join(set(string.punctuation) | set("।॥“”‘’…–—«»"))


def tokens(text):
    return [t.strip(EDGE).lower() for t in text.split() if t.strip(EDGE)]


def load_pool():
    train_counts = Counter(t for s in pd.read_csv(PROCESSED / "train.csv")["original"]
                           for t in tokens(s))
    thr = pd.read_csv(FINAL / "test_comparison.csv").set_index("model")["threshold"][BLEND]
    frames = []
    for split in ("val", "test"):
        df = pd.read_csv(PROCESSED / f"{split}.csv")
        feats = pd.concat([pd.read_csv(FEATURES / f"{g}_{split}.csv") for g in ("basic", "ne")], axis=1)
        pred = pd.read_csv(FINAL / f"{split}_all_predictions.csv")[BLEND]
        d = pd.concat([df[["original", "translation", "z_mean", "mean"]], feats], axis=1)
        d["pred"] = pred.to_numpy()
        d["split"] = split
        # rare-word proxy: share of source tokens seen at most once in the train sources
        d["rare_frac"] = [np.mean([train_counts[t] <= 1 for t in tokens(s)] or [0])
                          for s in df["original"]]
        frames.append(d)
    pool = pd.concat(frames, ignore_index=True)
    cutoff = load_cutoff()
    pool["fail"] = (pool["z_mean"] < cutoff).astype(int)
    pool["flagged"] = (pool["pred"] < thr).astype(int)
    pool["abs_err"] = (pool["pred"] - pool["z_mean"]).abs()
    return pool, cutoff, thr


def groups(p):
    rare_cut = p["rare_frac"].quantile(0.8)
    return {
        "Source length 3-10 words": p.src_words <= 10,
        "Source length 11-20 words": p.src_words.between(11, 20),
        "Source length 21-35 words": p.src_words.between(21, 35),
        "Source length 36+ words": p.src_words >= 36,
        "Contains numbers": p.src_num_count > 0,
        "Number mismatch (missing/extra)": (p.src_num_count > 0) & (p.all_numbers_match == 0),
        "No named entities": p.src_entities == 0,
        "1-2 named entities": p.src_entities.between(1, 2),
        "3+ named entities": p.src_entities >= 3,
        "Contains acronym": p.src_acronyms > 0,
        f"Rare words (top 20%, share >= {rare_cut:.2f})": p.rare_frac >= rare_cut,
        "English words left in translation": p.mt_latin_words > 0,
        "Translation much shorter (word ratio < 0.85)": p.word_ratio < 0.85,
        "Translation much longer (word ratio > 1.40)": p.word_ratio > 1.40,
        "Source has brackets": p.src_brackets > 0,
        "Source has quotes": p.src_quotes > 0,
        "Heavy punctuation (6+ marks)": p.src_punct >= 6,
        "No terminal punctuation (fragment)": p.src_ends_terminal == 0,
    }


def wilson(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def analyse(p):
    overall = p["fail"].mean()
    rows = []
    for name, mask in groups(p).items():
        g, rest = p[mask], p[~mask]
        k, n = int(g["fail"].sum()), len(g)
        k2, n2 = int(rest["fail"].sum()), len(rest)
        lo, hi = wilson(k, n)
        odds, pval = fisher_exact([[k, n - k], [k2, n2 - k2]])
        fails = g[g["fail"] == 1]
        rows.append({
            "group": name, "n": n, "share_of_data": n / len(p),
            "fail_rate": k / n, "fail_lo": lo, "fail_hi": hi,
            "fail_rate_rest": k2 / n2, "relative_risk": (k / n) / max(k2 / n2, 1e-9),
            "p_value": pval, "mean_z_mean": g["z_mean"].mean(),
            "model_mae": g["abs_err"].mean(),
            "model_recall": fails["flagged"].mean() if len(fails) else np.nan,
            "model_flag_rate": g["flagged"].mean(),
        })
    t = pd.DataFrame(rows)
    t["strong_after_bonferroni"] = t["p_value"] < 0.05 / len(t)
    return t, overall


def plot(t, overall):
    t = t.sort_values("fail_rate")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(t))
    ax.barh(y, t["fail_rate"], xerr=[t["fail_rate"] - t["fail_lo"], t["fail_hi"] - t["fail_rate"]],
            color="#4c78a8", capsize=2)
    ax.axvline(overall, color="#d62728", ls="--", label=f"overall {overall:.1%}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(t["group"], t["n"])], fontsize=8)
    ax.set_xlabel("share of pairs that fail (true score below the 15% cutoff), 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "failure_rate_by_type.png", dpi=150)
    plt.close(fig)


def worst_cases(p):
    cols = ["split", "original", "translation", "z_mean", "pred", "src_words", "mt_latin_words",
            "src_entities", "src_num_count", "all_numbers_match"]
    missed = p[(p.fail == 1) & (p.flagged == 0)].sort_values("pred", ascending=False).head(20)
    false_alarm = p[(p.fail == 0) & (p.flagged == 1)].sort_values("pred").head(20)
    missed[cols].to_csv(OUT / "missed_failures.csv", index=False)
    false_alarm[cols].to_csv(OUT / "false_alarms.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool, cutoff, thr = load_pool()
    print(f"{len(pool)} rows (val+test); fail cutoff {cutoff:.3f}; blend threshold {thr:.3f}")
    print(f"overall fail rate {pool.fail.mean():.1%}; model flags {pool.flagged.mean():.1%}; "
          f"recall {pool.loc[pool.fail == 1, 'flagged'].mean():.1%}; "
          f"precision {pool.loc[pool.flagged == 1, 'fail'].mean():.1%}")
    table, overall = analyse(pool)
    table.to_csv(OUT / "by_sentence_type.csv", index=False)
    plot(table, overall)
    worst_cases(pool)
    view = table.copy()
    for c in ("share_of_data", "fail_rate", "fail_rate_rest", "model_recall", "model_flag_rate"):
        view[c] = (view[c] * 100).round(1)
    view["fail_rate"] = [f"{r:.1f} [{lo*100:.0f}-{hi*100:.0f}]"
                         for r, lo, hi in zip(view["fail_rate"], table["fail_lo"], table["fail_hi"])]
    view["p_value"] = view["p_value"].map(lambda v: f"{v:.3f}" if v >= 0.001 else "<0.001")
    view["sig"] = np.where(table["strong_after_bonferroni"], "**",
                           np.where(table["p_value"] < 0.05, "*", ""))
    with pd.option_context("display.width", 250, "display.max_colwidth", 52):
        print(view[["group", "n", "fail_rate", "relative_risk", "p_value", "sig",
                    "model_mae", "model_recall", "model_flag_rate"]]
              .round(2).to_string(index=False))
    print("\n* p<0.05, ** p<0.05/18 (Bonferroni). fail_rate in % with 95% CI; "
          "relative_risk = fail rate in group / fail rate outside it.")


if __name__ == "__main__":
    main()
