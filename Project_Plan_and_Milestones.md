# Translation Confidence Scoring (English → Hindi MT Quality Estimation)

Course: Fall26 – Case Studies / Machine Learning (54530)
Timeline: 25 days, Day 1 = Wed 28 Oct 2026 → Day 25 = Sat 21 Nov 2026
Time budget: 2 hrs on weekdays, 3 hrs each on Sat and Sun (6 hrs per weekend) ≈ 57 hours total

---

## Special Instructions

### Where to run: PyCharm (local) + Google Colab (GPU only)
Develop locally in PyCharm; run only the GPU-heavy work on Colab (or Kaggle as fallback).

| Task | Where |
|---|---|
| Data cleaning, EDA, splits | PyCharm |
| Hand-crafted features (lengths, overlap, number/NE) | PyCharm |
| Perplexity and LaBSE embeddings | PyCharm; Colab if slow (large data). Cache to disk. |
| Linear, RF, XGBoost baselines | PyCharm |
| NLLB translation (backup data route only) | Colab |
| XLM-R fine-tuning | Colab |
| SHAP, error analysis, plots, prototype, report | PyCharm |

### Workflow
1. One project folder with `src/` scripts, `data/`, and `requirements.txt`.
2. Code in a private GitHub repo or synced via Google Drive. On Colab, clone/mount it, run the training script, and save checkpoint + predictions to Drive.
3. Download predictions/scores back to PyCharm for evaluation and analysis.
4. Write fine-tuning as a script with arguments (data path, learning rate, seed) so the Colab notebook is just a few cells calling it; a second config is then a one-line change.

### Caveats
- Free Colab gives no guaranteed GPU and may cut sessions after a few hours: save a checkpoint every epoch. Kaggle (~30 GPU hrs/week) works the same way as a fallback.
- Fine-tuning XLM-R-base on a Mac CPU/MPS would take many hours per run; use Colab for real runs. MPS is fine only for small experiments.
- Mac chip type not yet confirmed; it decides whether local perplexity/embedding runs are practical.

### Folder structure
```
translation_confidence_scoring/
├── src/          # scripts (preprocessing, features, train, evaluate)
├── data/raw/     # downloaded datasets
├── data/processed/  # cleaned data and splits
├── features/     # cached features, embeddings, perplexity
├── models/       # checkpoints and trained models
├── results/      # metrics, plots, tables
├── notebooks/    # EDA and Colab notebooks
├── config.yaml
├── requirements.txt
└── .gitignore
```

---

## 1. Project Plan

### 1.1 Goal
Build a model that predicts how reliable an English→Hindi machine translation is, using only the source sentence and the MT output (no reference). Output: a continuous confidence score (0–100) plus a binary "likely OK / likely to fail" label from a tuned threshold.

### 1.2 Research Questions (RQ)
1. How accurately can quality be predicted from source + MT only?
2. Do classical models (Linear Regression, Random Forest, XGBoost) with hand-crafted features match fine-tuned XLM-RoBERTa-base?
3. Which features are most informative of failure (length ratio, LM perplexity, embedding similarity, …)?
4. Which sentence types fail most (long, idioms, rare words, named entities, numbers)?
5. Can the score become a reliable "send to human review" threshold with acceptable precision/recall?

### 1.3 Data
- **Primary:** WMT QE / MLQE-PE English-Hindi (source, MT, DA score or HTER).
- **Backup:** Translate a public parallel corpus (Tatoeba / Multi30k subset) with a pretrained MT model (e.g. NLLB) and derive labels with chrF or COMET vs. the reference.
- **Decision gate (Day 3):** if MLQE-PE en-hi is not usable by end of Day 3, switch to the backup.

### 1.4 Pipeline
1. **Preprocessing** – clean, dedupe, drop empty/misaligned pairs, Unicode normalization, tokenize, normalize scores, train/val/test split (fixed seed; no leakage of the same source across splits).
2. **Features** – source/target length, length ratio, token & punctuation overlap, number and named-entity consistency, LM perplexity, LaBSE (or mBERT) cosine similarity.
3. **Models** – Baselines: Linear Regression, Random Forest, XGBoost. Advanced: fine-tuned XLM-RoBERTa-base with regression head on the (source, translation) pair.
4. **Prediction** – continuous score + binary fail label using threshold tuned on validation set.
5. **Evaluation** – Pearson, Spearman, RMSE, MAE; precision, recall, F1, ROC-AUC; error analysis; SHAP feature importance.

### 1.5 Deliverables
- Clean, reproducible code and a README
- Comparison table: baselines vs. XLM-R
- Analysis of what drives failure (RQ3 & RQ4) and threshold analysis (RQ5)
- Working prototype (command-line script) that flags low-confidence translations
- Final report (and slides if required)

### 1.6 Tools
Python, pandas, scikit-learn, XGBoost, Hugging Face Transformers, PyTorch, SHAP, Matplotlib/Seaborn.

### 1.7 Scope Decisions (to fit ~57 hours)
- **Dropped:** English-Spanish extension and Streamlit demo (revisit only if far ahead).
- **Capped:** hyperparameter tuning at 2–3 configurations per model.
- **Moved earlier:** XLM-R fine-tuning starts Day 11 (Sat 7 Nov) (Colab GPU runs in background; active time ≈ 2 hrs).
- **Never cut:** comparison table, SHAP, error analysis, threshold analysis.

### 1.8 Risks
| Risk | Mitigation |
|---|---|
| en-hi QE data unavailable/unusable | Backup route (Day 3 gate); if used, merge feature work on Days 5–8 to absorb the extra day |
| No GPU for XLM-R fine-tuning | Google Colab / Kaggle GPU; max_len 128, fp16, few epochs |
| Fine-tuning underperforms | Tune LR (1e-5–3e-5), try extra seeds, freeze lower layers |
| Little buffer (only Days 23 and 24) | Cut tuning first; Day 25 is submission day |
| Skewed labels (few failures) | Choose fail threshold by percentile/DA cutoff; report class balance |

### 1.9 Open Questions for the Instructor (from the proposal)
Send on Day 1:
1. Is one language pair sufficient?
2. Regression, classification, or both? (Plan covers both.)
3. Is pretrained XLM-RoBERTa-base acceptable?
4. Any expected evaluation standards/baselines?

---

## 2. Weekly Milestones

| Week | Days | Dates | Milestone (exit criteria) |
|---|---|---|---|
| **Week 1** | 1–7 | 28 Oct – 3 Nov | Environment ready, data secured and cleaned, splits fixed, simple/NE/number features done, perplexity features started |
| **Week 2** | 8–14 | 4 – 10 Nov | LaBSE features done; Linear/RF/XGBoost baselines evaluated; XLM-R fine-tuned and best config picked; final comparison table (RQ1, RQ2) |
| **Week 3** | 15–21 | 11 – 17 Nov | SHAP (RQ3), error analysis (RQ4), threshold analysis (RQ5), CLI prototype, report drafted, code clean-up |
| **Week 4** | 22–25 | 18 – 21 Nov | Proofreading, buffer, submission |

---

## 3. Daily Milestones

| Day | Date | Hrs | Task |
|---|---|---|---|
| 1 | Wed 28 Oct | 2 | Environment setup, Colab GPU check, email instructor's questions, find en-hi data |
| 2 | Thu 29 Oct | 2 | Download data, read docs (DA vs HTER), start exploratory analysis |
| 3 | Fri 30 Oct | 2 | **Decision gate:** WMT data or backup (start NLLB translation if backup) |
| 4 | Sat 31 Oct | 3 | Finish labels and exploratory analysis, preprocessing |
| 5 | Sun 1 Nov | 3 | Splits, "fail" label definition, length/overlap features |
| 6 | Mon 2 Nov | 2 | Number and named-entity consistency features |
| 7 | Tue 3 Nov | 2 | Perplexity features |
| 8 | Wed 4 Nov | 2 | LaBSE embeddings and cosine similarity (cache to disk) |
| 9 | Thu 5 Nov | 2 | Linear Regression and Random Forest baselines |
| 10 | Fri 6 Nov | 2 | XGBoost baseline, classification metrics |
| 11 | Sat 7 Nov | 3 | XLM-R script, dry run, launch first full run |
| 12 | Sun 8 Nov | 3 | Evaluate, adjust, launch second config |
| 13 | Mon 9 Nov | 2 | Pick best XLM-R config |
| 14 | Tue 10 Nov | 2 | Final test evaluation, comparison table (RQ1, RQ2) |
| 15 | Wed 11 Nov | 2 | SHAP and feature importance (RQ3) |
| 16 | Thu 12 Nov | 2 | Error analysis by sentence type (RQ4) |
| 17 | Fri 13 Nov | 2 | Threshold tuning (RQ5) |
| 18 | Sat 14 Nov | 3 | Command-line prototype, start report (methods) |
| 19 | Sun 15 Nov | 3 | Report: results, discussion, intro, conclusion |
| 20 | Mon 16 Nov | 2 | Finish report draft |
| 21 | Tue 17 Nov | 2 | Clean code, README, re-run check |
| 22 | Wed 18 Nov | 2 | Proofread, slides if required |
| 23 | Thu 19 Nov | 2 | Buffer |
| 24 | Fri 20 Nov | 2 | Buffer / final check |
| 25 | Sat 21 Nov | 3 | Final review and **submit** |

---

## 4. Rules of Thumb
- Fixed random seed and one config file for all experiments.
- Cache embeddings/perplexities; start fine-tuning no later than Day 11.
- Write report notes as each step finishes, not at the end.
- If you fall behind more than 1 day: cut tuning first, then use buffer days.

---

## 5. Action Items / Question Log
_Add follow-ups here as we work through individual tasks._

- [ ] Send guidance questions to the instructor (Day 1)
- [ ] Confirm en-hi data availability (Day 2–3)
