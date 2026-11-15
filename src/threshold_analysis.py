"""Threshold analysis for the "send to human review" flag (RQ5).

A pair is flagged for review if its predicted score is at or below a threshold.
Operating points are chosen on VALIDATION only and then applied unchanged to TEST,
so test numbers are honest. Test rows are never used to pick a threshold.

Operating points: max F1; recall targets (80%, 70%, 60%); precision target (60%);
fixed review budgets (flag the lowest-scored 10%, 20%, 30% of pairs).
Also: sensitivity to the fail definition (10% / 15% / 20% cutoffs, each re-tuned on
val) and the thresholds expressed on the 0-100 display scale. Outputs go to
results/threshold/. Run from the project root:

    python src/threshold_analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FINAL = ROOT / "results" / "final"
OUT = ROOT / "results" / "threshold"
BLEND = "blend: xgboost (all) + xlmr"
MODELS = [BLEND, "xgboost (all)"]
N_BOOT = 2000
SEED = 42


def flag_stats(y_fail, pred, t):
    flag = pred <= t
    tp = int((flag & (y_fail == 1)).sum())
    n_flag, n_fail = int(flag.sum()), int(y_fail.sum())
    precision = tp / n_flag if n_flag else np.nan
    recall = tp / n_fail if n_fail else np.nan
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    missed = n_fail - tp
    n_pass = len(y_fail) - n_flag
    return {"flag_rate": n_flag / len(y_fail), "precision": precision, "recall": recall,
            "f1": f1, "lift": precision / y_fail.mean() if n_flag else np.nan,
            "bad_in_passed": missed / n_pass if n_pass else np.nan}


def tune(y_fail, pred):
    """Thresholds (flag if pred <= t) chosen on validation data only."""
    prec, rec, thr = precision_recall_curve(y_fail, -pred)   # thr ascending in -pred
    t_pred = -thr                                            # descending in pred
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    out = {"max F1": t_pred[int(np.argmax(f1))]}
    for target in (0.8, 0.7, 0.6):                           # fewest flags reaching the recall
        ok = np.where(rec[:-1] >= target)[0]
        out[f"recall >= {int(target * 100)}%"] = t_pred[ok.max()] if len(ok) else np.nan
    ok = np.where(prec[:-1] >= 0.6)[0]                       # most recall reaching the precision
    out["precision >= 60%"] = t_pred[ok.min()] if len(ok) else np.nan
    for rate in (0.10, 0.20, 0.30):
        out[f"review budget {int(rate * 100)}%"] = float(np.quantile(pred, rate))
    return out


def bootstrap_ci(y_fail, pred, t, rng):
    n = len(y_fail)
    rows = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        rows.append(flag_stats(y_fail[i], pred[i], t))
    b = pd.DataFrame(rows)
    return {f"{m}_ci": tuple(np.nanpercentile(b[m], [2.5, 97.5]))
            for m in ("flag_rate", "precision", "recall")}


def z_to_display(train):
    """Linear map from z_mean to the raw 0-100 mean, fitted on train."""
    slope, intercept = np.polyfit(train["z_mean"], train["mean"], 1)
    return lambda z: intercept + slope * z


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cutoffs = json.loads((ROOT / "results" / "fail_cutoffs.json").read_text())["cutoffs"]
    val, test = (pd.read_csv(PROCESSED / f"{s}.csv") for s in ("val", "test"))
    pv, pt = (pd.read_csv(FINAL / f"{s}_all_predictions.csv") for s in ("val", "test"))
    to100 = z_to_display(pd.read_csv(PROCESSED / "train.csv"))
    rng = np.random.default_rng(SEED)

    # 1. operating points at the main (15%) cutoff
    main_cut = cutoffs["15"]["z_mean_cutoff"]
    rows = []
    for model in MODELS:
        yv, yt = (val["z_mean"].to_numpy() < main_cut).astype(int), (test["z_mean"].to_numpy() < main_cut).astype(int)
        for name, t in tune(yv, pv[model].to_numpy()).items():
            if np.isnan(t):
                rows.append({"model": model, "operating_point": name, "note": "not reachable on val"})
                continue
            v, s = flag_stats(yv, pv[model].to_numpy(), t), flag_stats(yt, pt[model].to_numpy(), t)
            ci = bootstrap_ci(yt, pt[model].to_numpy(), t, rng)
            rows.append({"model": model, "operating_point": name, "threshold_z": t,
                         "threshold_0_100": float(to100(t)),
                         **{f"val_{k}": x for k, x in v.items()},
                         **{f"test_{k}": x for k, x in s.items()},
                         **{f"test_{k}": x for k, x in ci.items()}})
    ops = pd.DataFrame(rows)
    ops.to_csv(OUT / "operating_points.csv", index=False)

    # 2. sensitivity to the fail definition (re-tuned on val for each cutoff)
    sens = []
    for p in ("10", "15", "20"):
        cut = cutoffs[p]["z_mean_cutoff"]
        yv, yt = (val["z_mean"].to_numpy() < cut).astype(int), (test["z_mean"].to_numpy() < cut).astype(int)
        for model in MODELS:
            t_all = tune(yv, pv[model].to_numpy())
            for name in ("max F1", "recall >= 70%", "review budget 20%"):
                t = t_all[name]
                s = flag_stats(yt, pt[model].to_numpy(), t)
                sens.append({"fail_percentile": int(p), "fail_rate_test": yt.mean(), "model": model,
                             "operating_point": name, "threshold_0_100": float(to100(t)),
                             "test_roc_auc": roc_auc_score(yt, -pt[model].to_numpy()), **{f"test_{k}": x for k, x in s.items()}})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "sensitivity_to_fail_definition.csv", index=False)

    # 3. review-workload table on test: share of bad translations caught per share reviewed
    yt = (test["z_mean"].to_numpy() < main_cut).astype(int)
    work = []
    for model in MODELS:
        pred = pt[model].to_numpy()
        for share in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            t = np.quantile(pred, share)
            s = flag_stats(yt, pred, t)
            work.append({"model": model, "review_share": share, "recall": s["recall"],
                         "precision": s["precision"], "lift": s["lift"],
                         "bad_in_passed": s["bad_in_passed"]})
    work = pd.DataFrame(work)
    work.to_csv(OUT / "review_workload_test.csv", index=False)

    # plots (test set, 15% cutoff)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for model in MODELS:
        pred = pt[model].to_numpy()
        p, r, _ = precision_recall_curve(yt, -pred)
        ax[0].plot(r, p, label=model)
        w = work[work.model == model]
        ax[1].plot(w.review_share, w.recall, marker="o", label=model)
    ax[0].axhline(yt.mean(), color="grey", ls="--", label=f"no skill ({yt.mean():.0%})")
    ax[0].set(xlabel="recall (share of bad translations caught)", ylabel="precision", title="Precision-recall (test)")
    ax[1].plot([0, 0.5], [0, 0.5], color="grey", ls="--", label="random review")
    ax[1].set(xlabel="share of translations sent to review", ylabel="share of bad translations caught",
              title="Review workload vs coverage (test)")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_curves_test.png", dpi=150)
    plt.close(fig)

    # console summary
    pd.options.display.width = 250
    pd.options.display.max_columns = None
    b = ops[ops.model == BLEND].copy()
    print(f"Fail = z_mean below {main_cut:.3f} (15% of train); test fail rate {yt.mean():.1%}\n")
    print("BLEND operating points (chosen on val, scored on test):")
    show = pd.DataFrame({
        "operating point": b.operating_point,
        "flag if score <=": b.threshold_0_100.round(1),
        "val prec/recall": [f"{p:.2f}/{r:.2f}" if pd.notna(p) else "-" for p, r in zip(b.val_precision, b.val_recall)],
        "test flagged": (b.test_flag_rate * 100).round(1).astype(str) + "%",
        "test precision": [f"{p:.2f} [{lo:.2f}-{hi:.2f}]" if pd.notna(p) else "-" for p, (lo, hi) in zip(b.test_precision, b.test_precision_ci)],
        "test recall": [f"{p:.2f} [{lo:.2f}-{hi:.2f}]" if pd.notna(p) else "-" for p, (lo, hi) in zip(b.test_recall, b.test_recall_ci)],
        "bad in passed": (b.test_bad_in_passed * 100).round(1).astype(str) + "%"})
    print(show.to_string(index=False))
    print("\nWorkload on test (BLEND): share reviewed -> recall / precision / bad left in passed set")
    print(work[work.model == BLEND].round(3).drop(columns="model").to_string(index=False))
    print("\nSensitivity to the fail definition (BLEND, max F1 re-tuned on val), test:")
    print(sens[(sens.model == BLEND) & (sens.operating_point == "max F1")]
          [["fail_percentile", "fail_rate_test", "threshold_0_100", "test_roc_auc", "test_flag_rate", "test_precision", "test_recall", "test_f1"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
