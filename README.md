# One-Year-Ahead Financial Distress Prediction on the Indonesia Stock Exchange

Reproducibility repository for the study comparing the **Altman Z″-Score** with
three machine learning models — **Logistic Regression, Random Forest, and
XGBoost** — for one-year-ahead financial-distress prediction among non-financial
companies listed on the Indonesia Stock Exchange (IDX), FY2021–FY2025, with
**SHAP**-based interpretation.

The repository contains the analysis code, the derived analytical dataset, and
the scripts that document how the raw financial statements were retrieved and
parsed. It reproduces every number, table, and figure reported in the paper.

---

## What this study does

- Builds a firm-year panel of accounting ratios from IDX-filed audited
  financial statements.
- Labels a firm distressed in year *t* if it reports **negative net income or
  negative total equity in year *t*+1** (defined independently of the
  Z″-Score to avoid circularity).
- Tunes each model with **firm-level grouped five-fold cross-validation** and
  **SMOTE applied within training folds only**, then evaluates once on an
  **untouched temporal hold-out** (FY2024 predictors → FY2025 outcomes,
  *n* = 748).
- Quantifies uncertainty with **2,000 paired bootstrap resamples** (percentile
  95% CIs) and assesses probability quality with a **calibration curve and
  Brier score**.
- Interprets the cross-validation-selected model (Random Forest) with
  **TreeExplainer SHAP**, globally and locally.
- Reports two **sensitivity analyses**: a persistent-distress label and
  alternative grey-zone treatments.

## Headline results (hold-out test set, FY2024 → FY2025, *n* = 748)

| Model | ROC-AUC | Recall | F1 | Precision |
|---|---|---|---|---|
| Altman Z″-Score (cutoff 1.1) | 0.684 | 0.456 | 0.531 | 0.635 |
| Logistic Regression | 0.890 | 0.793 | 0.722 | 0.669 |
| Random Forest (selected) | 0.894 | 0.756 | 0.731 | 0.729 |
| XGBoost | 0.895 | 0.756 | 0.729 | 0.707 |

Bootstrap 95% CIs for each model's ROC-AUC advantage over the Z″-Score exclude
zero. Global SHAP importance is led by profitability ratios (ROA, X₃ = EBIT/TA,
X₂ = RE/TA); the liquidity component X₁ = WC/TA — Altman's most heavily
weighted term — contributes least.

---

## Repository structure

```
idx-distress-prediction/
├── README.md                 this file
├── requirements.txt          pinned package versions
├── LICENSE                   MIT
├── data/
│   ├── panel_xlsx.csv        firm-year records parsed from .xlsx statements
│   ├── panel_xbrl.csv        records recovered from XBRL (39 corrupt .xlsx)
│   └── README_data.md        data provenance and what is / isn't shared
├── scripts/
│   ├── 00_download.py        retrieve raw statements from IDX (optional)
│   ├── 01a_parse_xlsx.py     .xlsx  → panel_xlsx.csv
│   ├── 01b_parse_xbrl.py     XBRL   → panel_xbrl.csv
│   └── 02_run_analysis.py    full analysis → results/ and figures/
├── results/                  generated CSVs (performance, CIs, SHAP)
└── figures/                  generated PNGs (ROC, calibration, SHAP)
```

## How to reproduce

The analysis runs entirely from the two derived files already in `data/`; you
do **not** need to re-download anything from IDX.

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cd scripts
python 02_run_analysis.py
```

Runtime is a few minutes on a laptop CPU. Outputs are written to `results/`
and `figures/`. All stochastic steps use random seed 42; with the pinned
package versions the published numbers reproduce exactly.

### Optional: rebuilding the data from source

The raw statements are public IDX property and are **not** redistributed here
(see `data/README_data.md`). To rebuild the derived files from scratch:

1. `00_download.py` — retrieves the `FinancialStatement-<year>-Tahunan-<code>.xlsx`
   workbooks (and, as a fallback, the `instance.zip` XBRL archives) from the
   IDX disclosure portal. The per-year report listings are first saved from a
   browser; the script then downloads the linked files. See the header of the
   script for the manual step and the exact endpoints.
2. `01a_parse_xlsx.py` / `01b_parse_xbrl.py` — parse those files into
   `panel_xlsx.csv` and `panel_xbrl.csv`.

## Data and methodology notes

- **Ratios only.** All predictors are ratios, so differences in reporting
  currency (a few issuers report in USD) and rounding (full rupiah vs
  millions) require no conversion — ratios are scale-invariant.
- **Balance-sheet items are end-of-fiscal-year values**, with no averaging
  across years (this includes all total-asset denominators, e.g. ROA).
- **Leverage** is total liabilities / total assets (book leverage).
- **Outcome** is an accounting-based proxy for financial distress, not a legal
  or regulatory classification.

## Citation

If you use this repository, please cite the associated paper (details to be
added on acceptance) and this repository.

## License

Code: MIT (see `LICENSE`). The derived data are released for research
reproducibility; the underlying raw financial statements remain the property
of the Indonesia Stock Exchange (www.idx.co.id).
