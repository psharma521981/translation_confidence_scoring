# Project Log: Steps and Decisions

Working log for the final write-up. Bullet points only; expand into paragraphs later.
Project: Translation Confidence Scoring (English → Hindi MT Quality Estimation).
Companion file: `Project_Plan_and_Milestones.md` (plan and schedule).

---

## 1. Setup and Environment (Day 1, 28 Oct 2026)
- Develop locally in PyCharm; use Google Colab (free T4 GPU) only for GPU-heavy work (XLM-R fine-tuning).
- Framework: **PyTorch** (with Hugging Face Transformers and Trainer). Keras considered and rejected: XLM-R support is PyTorch-first, most QE examples use PyTorch, and the plan already listed PyTorch.
- Local venv with Python 3.11 (default 3.14 avoided because of possible missing wheels for torch/shap).
- Machine: Apple M4 Max; PyTorch MPS available for small local experiments.
- XGBoost needed `brew install libomp` on macOS.
- Colab check passed: Tesla T4, CUDA available (torch 2.11.0+cu130), 20 matrix multiplies (4096×4096) in 0.82 s, Google Drive write works.
- Packages: pandas, numpy, scikit-learn, xgboost, torch, transformers, datasets, sentence-transformers, sacrebleu, shap, matplotlib, seaborn, tqdm, pyyaml.
- Reproducibility: single `config.yaml` with seed 42.

## 2. Data Sourcing
- Primary data found: WMT-style English-Hindi QE data with human direct-assessment (DA) scores. The backup route (translate a corpus + chrF/COMET labels) was **not needed**.
- Files used: `train.enhi.df.short.tsv` (7,000 rows) and `dev.enhi.df.short.tsv` (1,000 rows).
- Columns: `index`, `original` (English source), `translation` (Hindi MT output), `scores` (4 annotator ratings), `mean`, `z_scores`, `z_mean`.
- Unlabeled test files (`test.enhi.final.src/.mt`, 1,074 pairs, no scores) were **deleted**: unusable for supervised evaluation.
- A TSV from a different dataset (other language pair) was set aside: scores from another language cannot label English-Hindi.

## 3. Understanding the Labels
- `scores`: four raw 0–100 annotator ratings per translation.
- `mean`: plain average of the four raw scores.
- `z_scores`: each annotator's raw score standardized as (score − that annotator's average) / that annotator's standard deviation. Verified by recomputing on all 7,000 train rows (exact match).
- `z_mean`: average of the four z-scores.
- Why z-scores: annotators use the scale differently (average raw score by annotator: 88.5, 87.5, 72.9, 74.4; std 12.1, 11.9, 9.7, 10.7). Z-scores remove annotator bias.
- Raw score distribution (train): mean 80.8, median 82.0, std 7.9, range 14.2–98.2. About 7.5% of rows below 70; about 2.3% below 60.
- Dev distribution is very similar (mean 80.8, 8.6% below 70).

## 4. Data Quality Findings (raw)
- No missing values or empty strings; all translations contain Devanagari text.
- 762 repeated sources in train (all repeated as identical source+translation pairs); 16 in dev.
- **233 sources appear in both train and dev** (leakage if dev were used as test).
- Scores are skewed high, so failures are rare (class imbalance for the "fail" label).

## 5. Preprocessing (script: `src/preprocess.py`)
- Pool train and dev (8,000 rows) rather than keeping the original train/dev split. Reason: dev has 233 sources overlapping with train and is too small for separate validation and test sets.
- Unicode NFC normalization and whitespace stripping on source and translation.
- Empty rows removed: 0.
- **Deduplication rule:** drop exact repeats of (source, translation), keep the first. Chosen for simplicity and transparency; averaging repeated ratings was the alternative.
- Duplicates removed: 1,011 (778 inside train/dev + 233 cross-file). Clean rows: 6,989; every remaining source is unique.
- **Split:** grouped by source sentence (`GroupShuffleSplit`, seed 42, 10% test). A check in the script asserts no source appears in both splits.
- Result: `trainval.csv` 6,290 rows; `test.csv` 699 rows (fixed test set); `clean_all.csv` 6,989 rows (all cleaned data).
- Output files in `data/processed/`; run summary in `results/preprocess_summary.json`.

## 6. Evaluation Protocol Decisions
- Test set is created **once**, kept fixed, and **not used during training or tuning**. Used only after models and thresholds are final (planned Day 14 for the comparison table).
- Train/validation is done at runtime from `trainval.csv` (seeded split or cross-validation); validation is used for hyperparameters and the fail threshold.
- Dev is not used as a test set (it is a tuning set in WMT convention and overlapped with train).

## 7. Open Decisions (not final yet)
- **Target: CONFIRMED.** `z_mean` is the main regression target (bias-corrected); `mean` is used for the 0–100 display. Set in `config.yaml` (`target`, `display_score`).
- **"Fail" label:** proposed percentile cutoff on the training set only (10% / 15% / 20% as a sensitivity check; 15% as main). Estimated raw mean cutoff for 15% is about 73–74 (not yet computed).
  - Reason for percentile: failures are rare (about 8% below raw 70), which would leave too few positives for stable precision/recall.
  - Trade-off: wider cutoff includes mediocre translations in the "fail" class.
- Scores are not modified; the cutoff only affects the classification labels.

## 8. Next Steps
- Exploratory analysis: score distributions, lengths, annotator disagreement.
- Compute fail-cutoff values on the training portion and confirm the target.
- Feature engineering (Day 5 onward), then baselines, then XLM-R on Colab.

---

## Change Log
| Date | Change |
|---|---|
| 28 Oct | Environment, Colab check, data located |
| 29–30 Oct | Label columns analysed; raw data checked; split strategy decided |
| 30 Oct | `src/preprocess.py` run; processed files created |
