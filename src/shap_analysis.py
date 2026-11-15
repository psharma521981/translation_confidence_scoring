"""SHAP feature importance for the XGBoost model (all features).

The model is the config chosen on validation (results/val_comparison.csv), refit
on train only. SHAP values are computed on the VALIDATION rows (the test set is
not used for any analysis choices). Outputs go to results/shap/. Run from the
project root:

    python src/shap_analysis.py

Reading SHAP: for each prediction, a feature's SHAP value is how much it pushed
the predicted z_mean up (+) or down (-) from the average prediction.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import TARGET, load_features, models
from metrics import load_cutoff

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "shap"
GROUPS = ["basic", "ne", "labse", "ppl"]
GROUP_NAMES = {"basic": "length/punctuation/overlap", "ne": "numbers/entities",
               "labse": "LaBSE similarity", "ppl": "perplexity"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmp = pd.read_csv(ROOT / "results" / "val_comparison.csv")
    config = cmp[(cmp["model"] == "xgboost") & (cmp["features"] == "all")]["config"].iloc[0]
    est = dict(models()["xgboost"])[config]

    y_train = pd.read_csv(PROCESSED / "train.csv")[TARGET].to_numpy()
    y_val = pd.read_csv(PROCESSED / "val.csv")[TARGET].to_numpy()
    X_train, X_val = load_features("train", GROUPS), load_features("val", GROUPS)
    est.fit(X_train, y_train)
    print(f"xgboost config={config}, {X_train.shape[1]} features")

    explainer = shap.TreeExplainer(est)
    sv = explainer(X_val)
    values = sv.values

    # which file each feature came from, for the group summary
    owner = {}
    for g in GROUPS:
        for c in load_features("val", [g]).columns:
            owner[c] = g

    imp = pd.DataFrame({"feature": X_val.columns,
                        "mean_abs_shap": np.abs(values).mean(0),
                        "mean_shap_when_high": [
                            values[X_val[c] > X_val[c].median(), i].mean()
                            for i, c in enumerate(X_val.columns)],
                        "group": [owner[c] for c in X_val.columns]})
    # direction: rank correlation between the feature value and its SHAP value
    imp["direction_corr"] = [
        pd.Series(X_val[c].to_numpy()).corr(pd.Series(values[:, i]), method="spearman")
        for i, c in enumerate(X_val.columns)]
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["share_of_total"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    group_imp = (imp.groupby("group")["mean_abs_shap"].sum()
                 .rename("total_mean_abs_shap").to_frame())
    group_imp["share"] = group_imp["total_mean_abs_shap"] / group_imp["total_mean_abs_shap"].sum()
    group_imp["n_features"] = imp.groupby("group").size()
    group_imp = group_imp.sort_values("share", ascending=False)
    group_imp.to_csv(OUT / "group_importance.csv")

    # what pushes TRUE failures down (val rows below the fail cutoff)
    fail = y_val < load_cutoff()
    fail_push = pd.DataFrame({
        "feature": X_val.columns,
        "mean_shap_failures": values[fail].mean(0),
        "mean_shap_others": values[~fail].mean(0)})
    fail_push["gap"] = fail_push["mean_shap_failures"] - fail_push["mean_shap_others"]
    fail_push = fail_push.sort_values("gap").reset_index(drop=True)
    fail_push.to_csv(OUT / "failure_vs_ok_shap.csv", index=False)

    # plots
    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_bar_top15.png", dpi=150)
    plt.close()

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_beeswarm_top15.png", dpi=150)
    plt.close()

    top4 = list(imp["feature"][:4])
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, c in zip(axes, top4):
        shap.plots.scatter(sv[:, c], ax=ax, show=False)
    fig.tight_layout()
    fig.savefig(OUT / "shap_dependence_top4.png", dpi=150)
    plt.close(fig)

    summary = {
        "model": f"xgboost ({config}), all features, refit on train",
        "shap_computed_on": f"validation ({len(X_val)} rows)",
        "top10": imp.head(10)[["feature", "mean_abs_shap", "share_of_total",
                               "direction_corr"]].round(4).to_dict("records"),
        "groups": group_imp.round(4).to_dict("index"),
        "most_negative_for_failures": fail_push.head(8)[["feature", "gap"]].round(4).to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200):
        print("\nTop 15 features by mean |SHAP|:")
        print(imp.head(15)[["feature", "group", "mean_abs_shap", "share_of_total",
                            "direction_corr"]].round(3).to_string(index=False))
        print("\nBy feature group:")
        print(group_imp.round(3).to_string())
        print("\nLargest push-down for true failures vs others:")
        print(fail_push.head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
