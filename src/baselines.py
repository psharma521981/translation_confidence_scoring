"""Baseline models on the cached features: Linear Regression, Random Forest, XGBoost.

Trains on train.csv, evaluates on val.csv (test.csv is never loaded). Each model
is tried on two feature sets and, for RF/XGBoost, two small configurations; the
best configuration per model is picked by validation Pearson (tuning is capped
at 2 configurations, as planned). Run from the project root:

    python src/baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import best_f1_threshold, classification_metrics, load_cutoff, regression_metrics

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "features"
OUT = ROOT / "results" / "baselines"

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
SEED, TARGET = cfg["seed"], cfg["target"]

FEATURE_SETS = {
    "hand_crafted": ["basic", "ne"],
    "all": ["basic", "ne", "labse", "ppl"],
}


def load_features(split, groups):
    parts = [pd.read_csv(FEATURES / f"{g}_{split}.csv") for g in groups]
    X = pd.concat(parts, axis=1)
    return X.drop(columns=["mt_ppl"], errors="ignore")  # mt_nll carries the same information


def models():
    """name -> list of (config label, estimator)."""
    return {
        "linear": [("default", make_pipeline(StandardScaler(), LinearRegression()))],
        "random_forest": [
            (f"leaf{leaf}", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=leaf, max_features=0.5,
                n_jobs=-1, random_state=SEED))
            for leaf in (3, 10)
        ],
        "xgboost": [
            (f"depth{d}", XGBRegressor(
                n_estimators=400, learning_rate=0.03, max_depth=d, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, random_state=SEED, n_jobs=-1))
            for d in (3, 5)
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y_train = pd.read_csv(PROCESSED / "train.csv")[TARGET].to_numpy()
    y_val = pd.read_csv(PROCESSED / "val.csv")[TARGET].to_numpy()
    cutoff = load_cutoff()
    y_fail_val = (y_val < cutoff).astype(int)

    rows, best = [], {}
    for fs_name, groups in FEATURE_SETS.items():
        X_train, X_val = load_features("train", groups), load_features("val", groups)
        assert len(X_train) == len(y_train) and len(X_val) == len(y_val)
        for model_name, configs in models().items():
            for label, est in configs:
                est.fit(X_train, y_train)
                pred = est.predict(X_val)
                reg = regression_metrics(y_val, pred)
                key = (fs_name, model_name)
                if key not in best or reg["pearson"] > best[key]["reg"]["pearson"]:
                    best[key] = {"label": label, "pred": pred, "reg": reg,
                                 "n_features": X_train.shape[1]}
                print(f"{fs_name:13s} {model_name:14s} {label:8s} "
                      f"pearson={reg['pearson']:.3f} rmse={reg['rmse']:.3f}")

    for (fs_name, model_name), b in best.items():
        pred = b["pred"]
        t = best_f1_threshold(y_fail_val, pred)
        cls = classification_metrics(y_val, pred, cutoff)
        cls_tuned = classification_metrics(y_val, pred, cutoff, t)
        rows.append({
            "model": model_name, "features": fs_name, "config": b["label"],
            "n_features": b["n_features"], **b["reg"],
            "roc_auc": cls["roc_auc"], "pr_auc": cls["pr_auc"],
            "f1_at_label_cutoff": cls["f1"],
            "f1_tuned": cls_tuned["f1"], "precision_tuned": cls_tuned["precision"],
            "recall_tuned": cls_tuned["recall"], "threshold_tuned": t,
        })
        pd.DataFrame({"pred_z_mean": pred}).to_csv(
            OUT / f"{model_name}_{fs_name}_val_predictions.csv", index=False)

    table = pd.DataFrame(rows)
    # add fine-tuned runs that have a val_metrics.json (written by eval_predictions.py)
    for path in sorted((ROOT / "results").glob("xlmr*/val_metrics.json")):
        m = json.loads(path.read_text())
        t = m["fail_at_best_f1_threshold"]
        table.loc[len(table)] = {
            "model": path.parent.name, "features": "text pair", "config": "",
            **m["regression"], "roc_auc": m["fail_at_label_cutoff"]["roc_auc"],
            "pr_auc": m["fail_at_label_cutoff"]["pr_auc"],
            "f1_at_label_cutoff": m["fail_at_label_cutoff"]["f1"],
            "f1_tuned": t["f1"], "precision_tuned": t["precision"],
            "recall_tuned": t["recall"], "threshold_tuned": t["threshold"],
        }
    table = table.sort_values("pearson", ascending=False).reset_index(drop=True)
    table.to_csv(ROOT / "results" / "val_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print("\nValidation comparison (sorted by Pearson):")
        print(table.drop(columns=["threshold_tuned", "n_features"]))


if __name__ == "__main__":
    main()
