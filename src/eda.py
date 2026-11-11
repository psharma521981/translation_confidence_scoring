"""Exploratory analysis on trainval.csv only (the fixed test set is not touched).

Outputs: plots in results/eda/ and a summary in results/eda_summary.json.
"""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
out_dir = ROOT / "results" / "eda"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / cfg["data_dir"] / "processed" / "trainval.csv")
raw = df["scores"].apply(ast.literal_eval)
zs = df["z_scores"].apply(ast.literal_eval)
df["raw_std"] = raw.apply(np.std)
df["raw_range"] = raw.apply(lambda s: max(s) - min(s))
df["z_std"] = zs.apply(np.std)

df["src_chars"] = df["original"].str.len()
df["mt_chars"] = df["translation"].str.len()
df["src_words"] = df["original"].str.split().str.len()
df["mt_words"] = df["translation"].str.split().str.len()
df["len_ratio"] = df["mt_words"] / df["src_words"].clip(lower=1)

summary = {"rows": len(df)}

# 1. Score distributions
for col in ["mean", "z_mean"]:
    s = df[col]
    summary[col] = {
        "mean": s.mean(), "median": s.median(), "std": s.std(),
        "min": s.min(), "max": s.max(), "skew": s.skew(),
        "pct": {p: s.quantile(p / 100) for p in (5, 10, 15, 20, 25, 50, 75, 90)},
    }
summary["pct_mean_below_70"] = float((df["mean"] < 70).mean())
summary["pct_mean_below_60"] = float((df["mean"] < 60).mean())
summary["corr_mean_vs_z_mean_spearman"] = df["mean"].corr(df["z_mean"], method="spearman")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].hist(df["mean"], bins=50)
ax[0].set(title="Raw mean score", xlabel="mean (0-100)", ylabel="count")
ax[1].hist(df["z_mean"], bins=50)
ax[1].set(title="z_mean (target)", xlabel="z_mean")
fig.tight_layout()
fig.savefig(out_dir / "score_distributions.png", dpi=150)
plt.close(fig)

# 2. Lengths
len_cols = ["src_words", "mt_words", "len_ratio"]
summary["lengths"] = df[len_cols].describe().to_dict()
summary["length_vs_z_mean_spearman"] = {
    c: df[c].corr(df["z_mean"], method="spearman") for c in len_cols
}
df["src_len_bin"] = pd.qcut(df["src_words"], 5, duplicates="drop")
by_len = df.groupby("src_len_bin", observed=True)["z_mean"].agg(["count", "mean", "std"])
summary["z_mean_by_src_length_quintile"] = {
    str(k): v for k, v in by_len.to_dict("index").items()
}

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for a, c in zip(ax, len_cols):
    a.scatter(df[c], df["z_mean"], s=3, alpha=0.3)
    a.set(xlabel=c, ylabel="z_mean")
fig.tight_layout()
fig.savefig(out_dir / "length_vs_score.png", dpi=150)
plt.close(fig)

# 3. Annotator disagreement
summary["disagreement"] = {
    "raw_std_mean": df["raw_std"].mean(),
    "raw_range_median": df["raw_range"].median(),
    "pct_range_over_20": float((df["raw_range"] > 20).mean()),
    "z_std_mean": df["z_std"].mean(),
}
summary["disagreement_vs_mean_spearman"] = df["raw_std"].corr(df["mean"], method="spearman")
low = df["mean"] < df["mean"].quantile(0.15)
summary["raw_std_mean_low15"] = df.loc[low, "raw_std"].mean()
summary["raw_std_mean_rest"] = df.loc[~low, "raw_std"].mean()

# pairwise annotator agreement (z-scores, same row order across annotators)
z_mat = np.array(zs.tolist())
summary["annotator_z_corr"] = np.corrcoef(z_mat.T).round(3).tolist()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].hist(df["raw_range"], bins=40)
ax[0].set(title="Annotator range (max-min raw)", xlabel="points", ylabel="count")
ax[1].scatter(df["mean"], df["raw_std"], s=3, alpha=0.3)
ax[1].set(title="Disagreement vs quality", xlabel="mean", ylabel="std of 4 raw scores")
fig.tight_layout()
fig.savefig(out_dir / "annotator_disagreement.png", dpi=150)
plt.close(fig)

# 4. Lowest-scoring examples for a qualitative look
worst = df.nsmallest(15, "z_mean")[["original", "translation", "mean", "z_mean"]]
worst.to_csv(out_dir / "lowest_scored_examples.csv", index=False)

(ROOT / "results" / "eda_summary.json").write_text(
    json.dumps(summary, indent=2, default=float, ensure_ascii=False)
)
print(json.dumps(summary, indent=2, default=float, ensure_ascii=False))
