"""Shared evaluation metrics for regression and the "fail" classification.

A translation is "fail" if its true target is below the cutoff (from
results/fail_cutoffs.json, computed on train only). Lower predicted score means
more likely to fail, so the fail score for ROC-AUC is the negated prediction.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent


def load_cutoff(percentile=None):
    info = json.loads((ROOT / "results" / "fail_cutoffs.json").read_text())
    p = str(percentile or info["main_percentile"])
    return info["cutoffs"][p][f"{info['target']}_cutoff"]


def regression_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {"pearson": pearsonr(pred, y)[0], "spearman": spearmanr(pred, y)[0],
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae": float(np.mean(np.abs(pred - y)))}


def best_f1_threshold(y_fail, pred):
    """Prediction threshold (fail if pred < t) that maximises F1. Tune on val only."""
    prec, rec, thr = precision_recall_curve(y_fail, -np.asarray(pred))
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.argmax(f1[:-1]))
    return float(-thr[i])


def classification_metrics(y, pred, cutoff, threshold=None):
    """threshold: predictions below it are flagged fail. Default = the label cutoff."""
    y_fail = (np.asarray(y) < cutoff).astype(int)
    pred = np.asarray(pred)
    t = cutoff if threshold is None else threshold
    flag = (pred < t).astype(int)
    return {"threshold": float(t), "fail_rate_true": float(y_fail.mean()),
            "flagged_rate": float(flag.mean()),
            "precision": precision_score(y_fail, flag, zero_division=0),
            "recall": recall_score(y_fail, flag, zero_division=0),
            "f1": f1_score(y_fail, flag, zero_division=0),
            "roc_auc": roc_auc_score(y_fail, -pred),
            "pr_auc": average_precision_score(y_fail, -pred)}
