"""Final evaluation with bootstrap confidence intervals and paired comparisons.

Models: Linear / Random Forest / XGBoost on hand-crafted and on all features,
the fine-tuned XLM-R (saved predictions), and the 50/50 blend of XGBoost (all
features) with XLM-R. Feature models use the config chosen on validation
(results/val_comparison.csv, written by baselines.py) and are refit on train only.

    python src/baselines.py                      # refresh val_comparison.csv first
    python src/final_eval.py                     # validation only (safe, repeatable)
    python src/final_eval.py --test              # scores the test set ONCE

The classification threshold (fail if prediction < t) is tuned on validation and
applied unchanged to test. Validation numbers are therefore optimistic (tuned and
scored on the same rows); test numbers are the honest ones. --test refuses to run
again unless --force is given, so the test set is not used repeatedly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import TARGET, load_features, models
from metrics import best_f1_threshold, load_cutoff

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "final"

BASELINES = ["linear", "random_forest", "xgboost"]
FEATURE_GROUPS = {"hand_crafted": ["basic", "ne"], "all": ["basic", "ne", "labse", "ppl"]}
BLEND = "blend: xgboost (all) + xlmr"
METRICS = ["pearson", "spearman", "rmse", "mae", "roc_auc", "pr_auc", "precision", "recall", "f1"]
PAIRS = [
    (BLEND, "xgboost (all)"),
    (BLEND, "xlmr"),
    ("xgboost (all)", "xlmr"),
    ("xgboost (all)", "xgboost (hand_crafted)"),
    ("xlmr", "xgboost (hand_crafted)"),
]
DIFF_METRICS = ["pearson", "spearman", "rmse", "roc_auc", "pr_auc"]


def chosen_configs():
    table = pd.read_csv(ROOT / "results" / "val_comparison.csv")
    table = table[table["model"].isin(BASELINES)]
    return {(r.model, r.features): r.config for r in table.itertuples()}


def feature_model_predictions(configs, y_train, splits):
    """Refit each chosen config on train; return {name: {split: prediction}}."""
    out = {}
    for (model_name, fs), label in configs.items():
        est = dict(models()[model_name])[label]
        X_train = load_features("train", FEATURE_GROUPS[fs])
        est.fit(X_train, y_train)
        out[f"{model_name} ({fs})"] = {
            s: est.predict(load_features(s, FEATURE_GROUPS[fs])) for s in splits}
        print(f"fitted {model_name} ({fs}) config={label}")
    return out


def metric_values(y, pred, cutoff, thr):
    y_fail = (y < cutoff).astype(int)
    flag = pred < thr
    tp = int((flag & (y_fail == 1)).sum())
    prec = tp / max(int(flag.sum()), 1)
    rec = tp / max(int(y_fail.sum()), 1)
    return {
        "pearson": np.corrcoef(pred, y)[0, 1], "spearman": spearmanr(pred, y)[0],
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "roc_auc": roc_auc_score(y_fail, -pred), "pr_auc": average_precision_score(y_fail, -pred),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }


def bootstrap(y, preds, cutoff, thrs, n_boot, seed):
    """Same resampled rows for every model, so differences are paired."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    return {name: pd.DataFrame([metric_values(y[i], p[i], cutoff, thrs[name]) for i in idx])
            for name, p in preds.items()}


def summarise(y, preds, cutoff, thrs, boots):
    rows = []
    for name, p in preds.items():
        row = {"model": name, "threshold": thrs[name]}
        for m, v in metric_values(y, p, cutoff, thrs[name]).items():
            row[m] = v
            row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(boots[name][m], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pearson", ascending=False).reset_index(drop=True)


def paired_differences(boots, preds, y, cutoff, thrs):
    rows = []
    for a, b in PAIRS:
        va, vb = metric_values(y, preds[a], cutoff, thrs[a]), metric_values(y, preds[b], cutoff, thrs[b])
        for m in DIFF_METRICS:
            d = boots[a][m] - boots[b][m]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"a": a, "b": b, "metric": m, "diff_a_minus_b": va[m] - vb[m],
                         "lo": lo, "hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def show(table):
    with pd.option_context("display.width", 220, "display.max_columns", None):
        out = pd.DataFrame({"model": table["model"]})
        for m in ["pearson", "spearman", "rmse", "roc_auc", "pr_auc", "f1"]:
            out[m] = [f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
                      for v, lo, hi in zip(table[m], table[f"{m}_lo"], table[f"{m}_hi"])]
        print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="score the test set (once)")
    ap.add_argument("--force", action="store_true", help="allow scoring test again")
    ap.add_argument("--xlmr", default="results/xlmr_lr2e-05_seed42_ep3",
                    help="folder with val_predictions.csv and test_predictions.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    test_file = OUT / "test_comparison.csv"
    if a.test and test_file.exists() and not a.force:
        sys.exit(f"{test_file} exists: the test set was already scored. Use --force to redo.")

    splits = ["val", "test"] if a.test else ["val"]
    y = {s: pd.read_csv(PROCESSED / f"{s}.csv")[TARGET].to_numpy() for s in splits}
    y_train = pd.read_csv(PROCESSED / "train.csv")[TARGET].to_numpy()
    cutoff = load_cutoff()

    preds = feature_model_predictions(chosen_configs(), y_train, splits)
    xdir = ROOT / a.xlmr
    preds["xlmr"] = {}
    for s in splits:
        p = pd.read_csv(xdir / f"{s}_predictions.csv")["pred_z_mean"].to_numpy()
        assert len(p) == len(y[s]), f"xlmr {s} predictions do not match {s}.csv"
        preds["xlmr"][s] = p
    preds[BLEND] = {s: 0.5 * preds["xgboost (all)"][s] + 0.5 * preds["xlmr"][s] for s in splits}

    # thresholds are tuned on validation only
    thrs = {n: best_f1_threshold((y["val"] < cutoff).astype(int), p["val"]) for n, p in preds.items()}

    for s in splits:
        sp = {n: p[s] for n, p in preds.items()}
        boots = bootstrap(y[s], sp, cutoff, thrs, a.boot, a.seed)
        table = summarise(y[s], sp, cutoff, thrs, boots)
        diffs = paired_differences(boots, sp, y[s], cutoff, thrs)
        table.to_csv(OUT / f"{s}_comparison.csv", index=False)
        diffs.to_csv(OUT / f"{s}_paired_differences.csv", index=False)
        pd.DataFrame(sp).to_csv(OUT / f"{s}_all_predictions.csv", index=False)
        print(f"\n=== {s.upper()} ({len(y[s])} rows, {a.boot} bootstrap resamples, 95% CI) ===")
        show(table)
        print("\nPaired differences (a minus b); * = 95% CI excludes 0 (for rmse, negative is better for a):")
        d = diffs.copy()
        d["result"] = [f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if sig else "")
                       for x, lo, hi, sig in zip(d.diff_a_minus_b, d.lo, d.hi, d.ci_excludes_zero)]
        print(d.pivot_table(index=["a", "b"], columns="metric", values="result",
                            aggfunc="first").to_string())


if __name__ == "__main__":
    main()
