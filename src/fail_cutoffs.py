"""Split trainval into train/val and compute "fail" cutoffs on train only.

The split is grouped by source sentence (seeded). A translation is "fail" if its
target score is below the chosen percentile of the TRAIN scores. The same numeric
cutoff is then applied to val (and later to test); it is never recomputed there.
Run from the project root:

    python src/fail_cutoffs.py
"""
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    seed, val_size = cfg["seed"], cfg["val_size"]
    percentiles = cfg["fail_percentiles"]
    target, display = cfg["target"], cfg["display_score"]

    trainval = pd.read_csv(PROCESSED / "trainval.csv")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(trainval, groups=trainval["original"]))
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]
    assert not set(train["original"]) & set(val["original"]), "source leakage"

    train.to_csv(PROCESSED / "train.csv", index=False)
    val.to_csv(PROCESSED / "val.csv", index=False)

    out = {"seed": seed, "val_size": val_size, "rows_train": len(train),
           "rows_val": len(val), "target": target, "display_score": display,
           "main_percentile": cfg["fail_main_percentile"], "cutoffs": {}}
    for p in percentiles:
        q = p / 100
        cut_target = float(train[target].quantile(q))
        cut_display = float(train[display].quantile(q))
        out["cutoffs"][str(p)] = {
            f"{target}_cutoff": cut_target,
            f"{display}_cutoff": cut_display,
            "fail_rate_train": float((train[target] < cut_target).mean()),
            "fail_rate_val": float((val[target] < cut_target).mean()),
            "n_fail_train": int((train[target] < cut_target).sum()),
            "n_fail_val": int((val[target] < cut_target).sum()),
        }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fail_cutoffs.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
