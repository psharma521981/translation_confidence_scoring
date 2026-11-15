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
- **"Fail" label:** percentile cutoff on the train split only (10% / 15% / 20% as a sensitivity check; 15% as main). Computed by `src/fail_cutoffs.py` after a source-grouped train/val split of `trainval.csv` (val 15%, seed 42: 5,346 train / 944 val). Results in `results/fail_cutoffs.json`:
  - 10%: `z_mean` < −0.822 (raw mean < 71.5); 535 train / 98 val fails.
  - **15% (main):** `z_mean` < −0.581 (raw mean < 74.25); 802 train / 157 val fails (val rate 16.6%).
  - 20%: `z_mean` < −0.420 (raw mean < 76.25); 1,069 train / 204 val fails.
  - Same numeric cutoff is applied to val and test, never recomputed there. Files `train.csv` / `val.csv` saved in `data/processed/`.
  - Reason for percentile: failures are rare (about 8% below raw 70), which would leave too few positives for stable precision/recall.
  - Trade-off: wider cutoff includes mediocre translations in the "fail" class.
- Scores are not modified; the cutoff only affects the classification labels.

## 8. Basic Features (script: `src/features_basic.py`)
- 36 features per row, saved to `features/basic_{train,val,test}.csv` (same row order as `data/processed/{train,val,test}.csv`). Deterministic, nothing fitted, so the same code runs on all splits.
- Groups: length (chars, words, ratios, differences, average word length); punctuation (counts, differences, terminal punctuation, commas, quotes, brackets); script mix of the translation (Devanagari/Latin fraction, Latin words); cross-lingual overlap (tokens shared unchanged, copied Latin words); fluency proxies (type-token ratio, repeated tokens and bigrams).
- Tokenization: whitespace split with edge punctuation stripped (a `\w` regex would split Hindi words at vowel signs).
- Number and named-entity consistency are left for Day 6 (only raw digit counts here).
- Spearman with `z_mean` on train: all weak (|rho| ≤ 0.26). Strongest negatives: punctuation count (−0.26), source length (−0.24), Latin words in the translation (−0.21). Strongest positives: Devanagari fraction (+0.23), type-token ratio (+0.17). Length ratio is nearly useless (+0.04).
- Caveat: repeated-token counts partly track length (function words repeat in long sentences).

## 9. Number and Named-Entity Features (script: `src/features_ne.py`)
- 22 features, saved to `features/ne_{train,val,test}.csv` (same row order as the processed CSVs).
- **Numbers:** extracted from source and translation, Devanagari digits mapped to ASCII, thousands commas dropped (1,000 = 1000), ordinals (2nd, 21st) ignored because Hindi writes them as words. Compared as multisets: matched / missing / extra counts, match fraction, `all_numbers_match`.
- **Named entities (heuristic):** no NER library installed, so source entities = capitalised words that are not sentence-initial, plus all-caps acronyms; spans break at punctuation. Hindi output transliterates names into Devanagari, so exact matching only detects entities copied in Latin script. The rest are counted (`entities_not_copied`) but cannot be verified; treat these as complexity proxies, not true consistency checks.
- Other features: Latin spans in the translation, spans with no source counterpart, URLs lost.
- Dropped two constant features: Devanagari digits in the translation (always 0, translations use ASCII digits) and URLs kept.
- Findings on train: 16.5% of rows contain numbers; in 90% of those all numbers match. The 87 rows with a number mismatch average `z_mean` −0.45 versus −0.09 for matching rows, so this is a real but rare failure signal. Spearman correlations are weak (|rho| ≤ 0.20); the strongest is Latin spans in the translation (−0.20). URLs appear in only 4 train rows, so that feature is nearly empty.

## 10. Embedding, Perplexity and XLM-R Runs (Colab T4)
- **LaBSE cosine** (`features/labse_*.csv`) and **perplexity** with `facebook/xglm-564M` on the translation (`features/ppl_*.csv`: `mt_nll`, `mt_ppl`). Perplexity first ran out of GPU memory (256k vocabulary × batch 32); fixed with batch 8, length-sorted batches.
- Train Spearman with `z_mean`: `labse_cos` +0.06, `mt_nll` −0.07. Both weaker than expected and weaker than simple length/punctuation features.
- **XLM-R-base fine-tune** (lr 2e-5, seed 42, 3 epochs, max_len 128, batch 16, fp16, target `z_mean`). Issues: `warmup_ratio` removed in transformers 5 (use `warmup_steps=0.1`); Drive filled up with full checkpoints (3.3 GB each), so checkpoints now go to local disk with `save_only_model=True` and only the best model and predictions are copied to Drive.
- Validation (944 rows, `src/eval_predictions.py`): Pearson 0.525, Spearman 0.487, RMSE 0.648, MAE 0.448. Fail label (15% cutoff −0.581): ROC-AUC 0.753, PR-AUC 0.463. Using the label cutoff as the decision threshold flags only 4.4% of rows (precision 0.67, recall 0.18, F1 0.28); a threshold tuned for F1 on val (0.016) flags 34% (precision 0.34, recall 0.69, F1 0.46). The tuned numbers are optimistic because the threshold is picked on the same val set; final threshold analysis is Day 17.
- Per-epoch validation (Colab): epoch 1 Pearson 0.401 / Spearman 0.420 / val loss 0.511; epoch 2 0.492 / 0.469 / 0.441; epoch 3 0.525 / 0.487 / 0.420. Training loss at epoch 3 was 0.41, close to validation loss, so no overfitting; the model was still improving when training stopped (gains shrinking: +0.09 then +0.03). Colab's epoch-3 numbers match the ones recomputed from `val_predictions.csv`. Planned second config: same settings with 5 epochs (best epoch kept by `load_best_model_at_end`), run folder name to include `ep5`.

- **Update, reruns and run-to-run noise (13 Nov).** The 5-epoch run (same lr and seed) gave val Pearson 0.500, Spearman 0.489, RMSE 0.684, ROC-AUC 0.752: no better than 3 epochs, since the learning-rate schedule stretches over all epochs. The original 3-epoch predictions file (0.525) was lost (overwritten on Drive; a file restored from Trash turned out to be a different run, Pearson 0.468, and was set aside in `results/restored_unverified/`). Two reruns of the same 3-epoch config scored Pearson 0.504 and **0.520**. So the same config and seed gave 0.525, 0.504 and 0.520: single-run noise is about ±0.02 Pearson (GPU training is not exactly repeatable). XLM-R should be reported as about 0.50–0.52, and differences of 0.02 between runs (e.g. 3 vs 5 epochs) are not meaningful.
- **Chosen XLM-R run:** the last 3-epoch rerun in `results/xlmr_lr2e-05_seed42_ep3/` (val Pearson 0.520, Spearman 0.475, RMSE 0.641, MAE 0.467, ROC-AUC 0.734, PR-AUC 0.420, tuned F1 0.427). It is the only run with test predictions (`test_predictions.csv`, 699 rows, saved but not scored). Model weights are not stored (only predictions are kept on Drive, to avoid filling it). The 5-epoch predictions are kept in `results/xlmr_lr2e-05_seed42_ep5/`.
- Process lessons: write each run to a uniquely named folder (including epoch count); `eval_predictions.py` writes `val_metrics.json` next to the file it scores, so a wrong or overwritten file also overwrites the metrics; always save test predictions in the same session as the validation ones.
- Note for section 11: where the baseline table lists "XLM-R (3 epochs) 0.525", that is the lost original run; the current chosen run is 0.520.

## 11. Baselines (script: `src/baselines.py`, shared metrics in `src/metrics.py`)
- Linear Regression (standardised), Random Forest and XGBoost, trained on `train.csv` and scored on `val.csv`; test set never loaded. Two feature sets: hand-crafted (basic + number/entity, 58 features) and all (adds `labse_cos`, `mt_nll`; `mt_ppl` dropped as redundant with `mt_nll`). Tuning capped at 2 configs for RF (min leaf 3, 10) and XGBoost (depth 3, 5); best picked by val Pearson, so these numbers are slightly optimistic.
- Results (val, Pearson / Spearman / RMSE / ROC-AUC / tuned F1):
  - XGBoost, all features: 0.557 / 0.468 / 0.617 / 0.745 / 0.449
  - Random Forest, all: 0.544 / 0.470 / 0.624 / 0.735 / 0.462
  - Linear, all: 0.540 / 0.435 / 0.626 / 0.739 / 0.453
  - XLM-R (3 epochs): 0.525 / 0.487 / 0.648 / 0.753 / 0.457
  - XGBoost, hand-crafted only: 0.509 / 0.414 / 0.640 / 0.727 / 0.437
  - Random Forest, hand-crafted only: 0.491 / 0.404 / 0.648 / 0.725 / 0.431
  - Linear, hand-crafted only: 0.467 / 0.332 / 0.658 / 0.688 / 0.416
- Findings: weak features combine well (even Linear reaches 0.47 on hand-crafted). LaBSE and perplexity, near useless alone, add about +0.05 Pearson in combination. XLM-R has the best Spearman and ROC-AUC; XGBoost has the best Pearson, RMSE and PR-AUC. No model clearly dominates and gaps are small on 944 rows, with no significance test yet, so the claim is "comparable", not "classical beats XLM-R".
- Caveats: XLM-R run 1 was under-trained; the test set decides the final comparison.
- Outputs: `results/val_comparison.csv`, `results/baselines/*_val_predictions.csv`.
- Ideas: add the XLM-R prediction as a feature to XGBoost (stacking); bootstrap confidence intervals for the comparison table (Day 14).

## 12. Blending Feature Models with XLM-R (script: `src/blend.py`)
- Simple average of a feature-model prediction and the XLM-R prediction, both on the `z_mean` scale. No retraining and no tuning: 50/50 is reported; a weight sweep is printed for information only (picking the weight on val would make val scores optimistic). Stacking with the XLM-R prediction as an XGBoost feature was rejected for now: XLM-R train-set predictions are overfitted, so out-of-fold predictions (several more Colab runs) would be needed.
- Validation results, chosen XLM-R run (Pearson / Spearman / RMSE / ROC-AUC / PR-AUC / tuned F1):
  - XLM-R alone: 0.520 / 0.475 / 0.641 / 0.734 / 0.420 / 0.427
  - XGBoost (all features) alone: 0.557 / 0.468 / 0.617 / 0.745 / 0.496 / 0.449
  - **XGBoost + XLM-R, 50/50: 0.590 / 0.517 / 0.610 / 0.762 / 0.500 / 0.478**
  - Random Forest + XLM-R: 0.583 / 0.513 / 0.615 / 0.752 / 0.496 / 0.471
  - Linear + XLM-R: 0.578 / 0.511 / 0.616 / 0.763 / 0.484 / 0.459
- Robustness: the XGBoost + XLM-R blend scored Pearson 0.583–0.590 with each of the three scored XLM-R runs (0.504, 0.520, 0.500 alone; the weaker restored 0.468 file gave 0.572), a gain of about +0.03 over XGBoost alone. The gain is stable across XLM-R runs even though the XLM-R scores are noisy. Predictions of the two models correlate at about 0.62–0.68, so they carry complementary information (useful for RQ2). Best XLM-R weight is about 0.3–0.5, so 50/50 needs no tuning.
- Caveats: the feature models' configs were selected on val; no confidence intervals yet (bootstrap planned); everything is on val, and the test set decides the final claim.

## 13. Final Evaluation (script: `src/final_eval.py`; outputs in `results/final/`)
- Models: Linear / Random Forest / XGBoost on hand-crafted and on all features (configs chosen on val, refit on train only), the chosen XLM-R run (saved predictions), and the 50/50 blend of XGBoost (all) and XLM-R. Model set fixed before the test run.
- Method: 2,000 bootstrap resamples, 95% percentile CIs; the same resampled rows for every model so differences are paired. Fail threshold (fail if prediction < t) tuned for F1 on val and applied unchanged to test. Validation numbers are therefore optimistic for the classification metrics; test numbers are the honest ones.
- Guard: `--test` refuses to run again unless `--force` is given. **The test set was scored once (13 Nov) and must not be used for further model selection**; any later changes need a new held-out set or must be reported as exploratory.
- **Validation (944 rows), Pearson [95% CI]:** blend 0.590 [0.51, 0.66]; XGBoost (all) 0.557 [0.48, 0.63]; RF (all) 0.544; Linear (all) 0.540; XLM-R 0.520 [0.43, 0.60]; XGBoost (hand-crafted) 0.509. On val the blend beat XGBoost on Pearson (+0.032, CI excludes 0) and Spearman (+0.049).
- **Test (699 rows), Pearson / Spearman / RMSE / ROC-AUC / PR-AUC / tuned-threshold F1:**
  - Blend (XGBoost + XLM-R): 0.580 [0.51, 0.64] / 0.489 / 0.581 / 0.766 / 0.493 / 0.464
  - Random Forest (all): 0.579 [0.49, 0.66] / 0.464 / 0.575 / 0.761 / 0.509 / 0.457
  - XGBoost (all): 0.560 [0.47, 0.64] / 0.442 / 0.583 / 0.750 / 0.481 / 0.455
  - RF (hand-crafted): 0.532 / 0.415 / 0.596 / 0.721 / 0.472 / 0.406
  - XGBoost (hand-crafted): 0.528 / 0.397 / 0.598 / 0.703 / 0.440 / 0.380
  - Linear (all): 0.524 / 0.408 / 0.600 / 0.743 / 0.454 / 0.421
  - XLM-R: 0.498 [0.43, 0.56] / 0.456 / 0.615 / 0.725 / 0.436 / 0.395
  - Linear (hand-crafted): 0.470 / 0.325 / 0.622 / 0.678 / 0.411 / 0.402
- Test scores are close to val (blend 0.580 vs 0.590 Pearson), so no sign of overfitting to the validation set; the model ranking mostly held.
- **Paired differences on test (CI excludes 0 unless stated):**
  - Blend vs XLM-R: Pearson +0.082, RMSE −0.034, ROC-AUC +0.042, PR-AUC +0.057; solid.
  - Blend vs XGBoost (all): Spearman +0.047 only. Pearson +0.020 [−0.010, +0.053], RMSE and ROC-AUC not significant. The val Pearson gain did **not** replicate significantly on test: report the blend as improving rank correlation, with a smaller, uncertain Pearson gain.
  - Random Forest (all) ties the blend on Pearson (0.579 vs 0.580).
  - XGBoost (all) vs XLM-R: Pearson +0.062 [−0.015, +0.132], Spearman −0.014, ROC-AUC +0.026, none significant: **comparable** (RQ2). XLM-R's Pearson CI is narrower than the feature models'.
  - All features vs hand-crafted only (XGBoost): Pearson +0.032, RMSE −0.015, ROC-AUC +0.047, Spearman +0.045: LaBSE and perplexity add real value in combination.
  - XLM-R vs XGBoost (hand-crafted): only Spearman differs (+0.059); otherwise comparable.
- **Fail detection (RQ5 preview, test):** ROC-AUC 0.73–0.77 for the stronger models; F1 at the val-tuned threshold 0.40–0.46. Blend threshold −0.257 gives precision 0.455 and recall 0.472; XLM-R alone (threshold −0.141) gives precision 0.299 and recall 0.583. Threshold study (Day 17) still to do.
- Suggested wording for the report: XLM-R fine-tuning and a feature-based gradient-boosted model are comparable at predicting quality here (Pearson about 0.50–0.56 on test); combining them improves rank correlation; labels are noisy (annotators differ by about 10 points) which caps achievable correlation.

## 14. SHAP Feature Importance, RQ3 (script: `src/shap_analysis.py`; outputs in `results/shap/`)
- Model: XGBoost (all 60 features, depth 5), refit on train only; SHAP (TreeExplainer) computed on the **validation** rows, so the test set is not used for analysis choices. Outputs: beeswarm, bar chart and dependence plots, `feature_importance.csv`, `group_importance.csv`, `failure_vs_ok_shap.csv`, `summary.json`.
- **Top features (share of total mean |SHAP|):** perplexity `mt_nll` 11.2% (high perplexity lowers the prediction), `labse_cos` 10.8% (high similarity raises it), `mt_devanagari_frac` 7.0% (more Devanagari raises it), `mt_chars` 6.3% (longer translation lowers it), `src_punct` 5.6% (more punctuation lowers it).
- LaBSE and perplexity each had raw Spearman of only about ±0.07 with the score, yet are the top two inside the model: they work by correcting what length and punctuation cannot explain. Consistent with the +0.03 to +0.05 Pearson gain from adding them.
- **By group:** length/punctuation/overlap (36 features) 72% (spread over many correlated features); perplexity 11% (1 feature); LaBSE 11% (1 feature); numbers/entities (22 features) only 5.7%. The number/entity features took the most effort and add the least; the entity side is only a capitalisation heuristic.
- **What pushes true failures down** (val rows below the fail cutoff vs others): English (Latin) words left in the translation (`mt_latin_word_frac`; a few rows pushed down by over 1.0 on the `z_mean` scale), a translation much shorter than the source (low `word_diff`, probably omissions), extreme character/word length ratios, then low LaBSE similarity and high perplexity.
- Length ratio had raw Spearman 0.04 and looked useless, but SHAP shows it matters at the extremes; a straight-line correlation misses this.
- Caveats: correlated features (the length family) share credit, so individual ranks inside that family are unstable; SHAP explains the model, not causes.

## 15. Error Analysis by Sentence Type, RQ4 (script: `src/error_analysis.py`; outputs in `results/error_analysis/`)
- Pooled validation and test rows (1,643 pairs) with the blend's predictions; descriptive only, no model, config or threshold changed (the blend threshold was tuned on val, so recall on the val part is slightly optimistic). Failure rate per sentence type with Wilson 95% CIs, Fisher exact test against the rest, Bonferroni over 18 groups. Overall: 16.1% fail; the blend flags 16.4%, recall 47.5%, precision 46.8%.
- **Highest failure rates (fail rate; times the rate outside the group; ** = significant after Bonferroni):** number mismatch 42.9% (2.7x, **, only 28 rows); translation much shorter than the source (word ratio < 0.85) 38.9% (2.6x, **); English words left in the translation 36.3% (2.7x, **); heavy punctuation (6+ marks) 28.9% (2.2x, **); source without final punctuation (fragments) 27.3% (1.8x, **); source has brackets 26.6% (1.8x, **); rare words (top 20% share of source tokens seen at most once in train) 25.7% (1.9x, **); 3+ named entities 25.2% (1.9x, **); quotes 24.5% (1.7x, **); 36+ word sources 21.5% (1.4x, p<0.05 only). Lowest: no named entities 12.7% and 11-20 word sources 12.3% (both 0.7x). Shortest sources (3-10 words) are not worse than average.
- Consistent with SHAP: omissions (missing number, much shorter translation) and untranslated English words are the clearest failure types. Entities and rare words matter for failure rates even though the entity features add little to the model.
- **Where the model errs:** recall is only 22.6% for 3-10 word sources and 31% for sentences with no entities (flag rate about 9%): failures in short, simple sentences look unsuspicious. Recall is about 85% for Latin words left, much shorter translations and number mismatches. Recall depends on flag rate (complex sentences are flagged more), so it partly reflects that.
- Caveats: groups overlap (long sentences also have more punctuation, entities and rare words), so these are one-factor comparisons and partly reflect length (a logistic regression would separate them); the number-mismatch group is small; **idioms were not measured** (no idiom list); rare words are a proxy; label noise is highest where failures are.
- Lists for the qualitative section: `missed_failures.csv` (true failures the model passed) and `false_alarms.csv`.

## 16. Threshold Analysis, RQ5 (script: `src/threshold_analysis.py`; outputs in `results/threshold/`)
- Flag a pair for human review if its predicted score is at or below a threshold. **Operating points are chosen on validation only and applied unchanged to test** (test never used to pick a threshold). Fail = true `z_mean` below the 15% cutoff (-0.581); test fail rate 15.5%. Test numbers carry 2,000-resample bootstrap 95% CIs. Models: the blend (headline) and XGBoost alone (in the CSVs).
- Thresholds on the 0-100 display scale come from a straight-line fit of raw `mean` on `z_mean` (train rows), so they are approximate.
- **Review workload on test (blend), share reviewed: recall / precision / bad left in passed set:** 5%: 21% / 66% / 12.8%; 10%: 37% / 57% / 10.8%; 20%: 54% / 41% / 8.9%; 30%: 66% / 34% / 7.6%; 50%: 80% / 25% / 6.3%. With no review 15.5% of the set is bad. Lift over random review: 4.3x at 5%, 2.7x at 20%, 1.6x at 50%.
- **Named operating points (flag if 0-100 score <=):** max F1 77.9 (flags 16%, precision 0.46, recall 0.47); recall >= 80% on val: 81.0 (flags 52%, precision 0.24, recall 0.81 on test); recall >= 70%: 80.2 (flags 40%, precision 0.28, recall 0.71); precision >= 60%: 75.7 (flags 7%, precision 0.59, recall 0.27); review budgets 10/20/30%: thresholds 76.6 / 78.5 / 79.6.
- Thresholds tuned on val transferred reasonably to test (e.g. recall target 80% gave 0.81), but val has only 944 rows, so expect about +/-0.1 on any single number (see CIs).
- **Sensitivity to the fail definition** (max-F1 threshold re-tuned on val for each cutoff): test ROC-AUC 0.777 / 0.766 / 0.771 for the 10% / 15% / 20% definitions, so ranking quality does not depend on the choice; max-F1 precision 0.30 / 0.46 / 0.39, recall 0.49 / 0.47 / 0.65, F1 0.37 / 0.46 / 0.49. Precision drops as "fail" gets stricter because true failures are rarer.
- **Answer to RQ5:** the score works as a triage tool, not an automatic pass/fail: it ranks well (the lowest 5% are 66% truly bad versus 15.5% at random), but no threshold is both precise and complete (catching 80% of bad translations needs reviewing about half of everything, and about 6% of the passed set is still bad). Best used as a prioritised review queue ("review the lowest 20%" catches about half of the bad translations).
- Caveats: the review-budget rows use the lowest-scoring share of the test set itself; the prototype cannot use the blend (XLM-R weights not stored), so its thresholds would come from the XGBoost-only rows.

## 17. Next Steps
- Command-line prototype, README, report. Optionally refresh `results/val_comparison.csv` (re-run `src/baselines.py`) if models change.

---

## Change Log
| Date | Change |
|---|---|
| 28 Oct | Environment, Colab check, data located |
| 29–30 Oct | Label columns analysed; raw data checked; split strategy decided |
| 30 Oct | `src/preprocess.py` run; processed files created |
| 10 Nov | Day 4: target confirmed (`z_mean`, `mean` for display); EDA run (`src/eda.py`); fail cutoffs computed on train split (`src/fail_cutoffs.py`), `train.csv` / `val.csv` created |
| 10 Nov | Day 5: length, punctuation, script and overlap features (`src/features_basic.py`, 36 features) |
| 10 Nov | Day 6: number and named-entity features (`src/features_ne.py`, 22 features) |
| 10 Nov | Day 7: perplexity feature (XGLM-564M on the translation, Colab); first run hit GPU out-of-memory, fixed with batch 8 |
| 10 Nov | Day 8: LaBSE cosine similarity feature (Colab) |
| 11 Nov | Days 11–12: XLM-R-base fine-tune run 1 (lr 2e-5, seed 42, 3 epochs): val Pearson 0.525; fixed `warmup_ratio` and Drive-space issues; added `src/metrics.py`, `src/eval_predictions.py` |
| 11 Nov | Days 9–10: baselines (`src/baselines.py`): best val Pearson 0.557 (XGBoost, all features); comparison table in `results/val_comparison.csv` |
| 13 Nov | XLM-R 5-epoch run (val Pearson 0.500, no better than 3 epochs); original 3-epoch predictions lost; two 3-epoch reruns scored 0.504 and 0.520, showing about ±0.02 run-to-run noise |
| 13 Nov | Final XLM-R run chosen (3 epochs, val Pearson 0.520) with val and test predictions saved; weights not stored |
| 13 Nov | Blend (`src/blend.py`): XGBoost + XLM-R 50/50 reaches val Pearson 0.590 (vs 0.557 XGBoost, 0.520 XLM-R), stable across XLM-R runs |
| 13 Nov | Day 14: `src/final_eval.py` built (bootstrap CIs, paired differences); validation run, then **test set scored once**: blend Pearson 0.580, RF (all) 0.579, XGBoost (all) 0.560, XLM-R 0.498; blend beats XLM-R significantly but not XGBoost on Pearson (comparison table, RQ1, RQ2) |
| 14 Nov | Day 15: SHAP on XGBoost (`src/shap_analysis.py`, RQ3): perplexity and LaBSE are the top two features; numbers/entities contribute only 5.7%; Latin words left in the translation and very short translations are the strongest failure signals |
| 14 Nov | Day 16: error analysis by sentence type (`src/error_analysis.py`, RQ4): omissions, untranslated English words, heavy punctuation, entities and rare words fail most; model recall is low on short simple sentences (22.6%)
| 14 Nov | Day 17: threshold analysis (`src/threshold_analysis.py`, RQ5): thresholds chosen on val, scored on test; reviewing the lowest 20% catches 54% of bad translations (2.7x random), 80% recall needs reviewing about half; ROC-AUC about 0.77 whatever the fail definition |
