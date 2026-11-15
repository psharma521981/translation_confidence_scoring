"""Blend a baseline model's validation predictions with XLM-R's.

    python src/blend.py results/xlmr_lr2e-05_seed42_ep5/val_predictions.csv

Reports the 50/50 average (no tuning) and a weight sweep for information only:
picking the weight on val would make val scores optimistic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import best_f1_threshold, classification_metrics, load_cutoff, regression_metrics

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ["xgboost_all", "random_forest_all", "linear_all"]


def score(y, pred, cutoff):
    reg = regression_metrics(y, pred)
    t = best_f1_threshold((y < cutoff).astype(int), pred)
    cls = classification_metrics(y, pred, cutoff, t)
    return {"pearson": reg["pearson"], "spearman": reg["spearman"], "rmse": reg["rmse"],
            "mae": reg["mae"], "roc_auc": cls["roc_auc"], "pr_auc": cls["pr_auc"],
            "f1_tuned": cls["f1"]}


def main(xlmr_path):
    y = pd.read_csv(ROOT / "data" / "processed" / "val.csv")["z_mean"].to_numpy()
    cutoff = load_cutoff()
    xlmr = pd.read_csv(xlmr_path)["pred_z_mean"].to_numpy()
    rows = [{"model": "xlmr alone", **score(y, xlmr, cutoff)}]
    for name in BASELINES:
        base = pd.read_csv(ROOT / "results" / "baselines" / f"{name}_val_predictions.csv")["pred_z_mean"].to_numpy()
        rows.append({"model": f"{name} alone", **score(y, base, cutoff)})
        rows.append({"model": f"{name} + xlmr (50/50)", **score(y, 0.5 * base + 0.5 * xlmr, cutoff)})
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))

    base = pd.read_csv(ROOT / "results" / "baselines" / "xgboost_all_val_predictions.csv")["pred_z_mean"].to_numpy()
    print("\nWeight sweep, xgboost_all (w = weight on XLM-R), Pearson / RMSE:")
    for w in np.arange(0, 1.01, 0.1):
        s = regression_metrics(y, w * xlmr + (1 - w) * base)
        print(f"  w={w:.1f}  pearson={s['pearson']:.3f}  rmse={s['rmse']:.3f}")
    print("\ncorrelation between the two models' predictions:",
          round(float(np.corrcoef(base, xlmr)[0, 1]), 3))
    out = Path(xlmr_path).parent / "blend_val_metrics.csv"
    table.to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1])
