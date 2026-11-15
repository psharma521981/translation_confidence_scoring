"""Score a saved validation-prediction file.

    python src/eval_predictions.py results/xlmr_lr2e-05_seed42_ep3/val_predictions.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import best_f1_threshold, classification_metrics, load_cutoff, regression_metrics

ROOT = Path(__file__).resolve().parent.parent


def main(path):
    path = Path(path)
    pred = pd.read_csv(path)["pred_z_mean"].to_numpy()
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    assert len(pred) == len(val), "prediction file does not match val.csv"
    y, cutoff = val["z_mean"].to_numpy(), load_cutoff()

    out = {"file": str(path), "n": len(val), "fail_cutoff": cutoff,
           "regression": regression_metrics(y, pred),
           "fail_at_label_cutoff": classification_metrics(y, pred, cutoff)}
    t = best_f1_threshold((y < cutoff).astype(int), pred)
    out["fail_at_best_f1_threshold"] = classification_metrics(y, pred, cutoff, t)
    (path.parent / "val_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
