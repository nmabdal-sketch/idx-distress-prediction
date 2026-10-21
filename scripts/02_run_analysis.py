#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_analysis.py
==================
Reproduces the main analysis of the study comparing the Altman Z"-Score with
Logistic Regression, Random Forest, and XGBoost for one-year-ahead financial
distress prediction among non-financial firms listed on the Indonesia Stock
Exchange (FY2021-FY2025).

Inputs  (in ../data/):
    panel_xlsx.csv   firm-year records parsed from the .xlsx statements
    panel_xbrl.csv   firm-year records recovered from XBRL instance archives
                     (for issuer-years whose .xlsx files were corrupt at source)

Outputs (in ../results/ and ../figures/):
    results_primary.csv        RQ1 performance table (primary t->t+1 label)
    results_sensitivity.csv    persistent-distress sensitivity results
    bootstrap_ci.csv           95% bootstrap CIs for AUC and AUC differences
    shap_ranking.csv           global SHAP feature ranking (selected RF)
    fig_roc_metrics.png        RQ1 ROC curves + metric comparison
    fig_calibration.png        calibration curve (selected RF)
    fig_shap.png               SHAP summary + global importance
    fig_shap_local.png         local SHAP explanation (median true positive)

Usage:
    python 02_run_analysis.py

All stochastic components use random seed 42. Package versions that produced
the published numbers are pinned in ../requirements.txt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             roc_curve)
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings("ignore")

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

FEATURES = ["X1", "X2", "X3", "X4", "ROA", "leverage", "current_ratio"]
FIN_KEYWORDS = ["Keuangan", "Asuransi", "Pembiayaan", "Sekuritas"]


# --------------------------------------------------------------------------- #
# 1. Load and merge the two parser outputs into one firm-year table
# --------------------------------------------------------------------------- #
def load_panel():
    main = pd.read_csv(os.path.join(DATA, "panel_xlsx.csv"))
    xbrl = pd.read_csv(os.path.join(DATA, "panel_xbrl.csv"))
    keys = set(zip(main["kode"], main["tahun"]))
    add = xbrl[~xbrl.apply(lambda r: (r["kode"], r["tahun"]) in keys, axis=1)]
    panel = pd.concat([main, add], ignore_index=True)
    for c in FEATURES + ["Z2_score", "laba_bersih", "total_ekuitas"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["tahun"] = panel["tahun"].astype(int)
    return panel


def is_financial(row):
    iu = str(row.get("industri_utama") or "")
    sec = str(row.get("sektor") or "")
    return (any(k in iu for k in FIN_KEYWORDS)
            or sec.startswith("G. Financial") or sec.startswith("8. Finance"))


# --------------------------------------------------------------------------- #
# 2. Build the t -> t+1 prediction panel with a chosen outcome definition
# --------------------------------------------------------------------------- #
def build_panel(panel, definition):
    """definition: 'oneyear' (primary) or 'persistent' (sensitivity).

    The analytical set comprises non-financial firm-years with a computable
    Z"-Score (3,629 observations). Both the predictor year and the outcome
    year are drawn from this set, so that benchmark and machine learning
    models are compared on identically constructed observations.
    """
    analytic = panel[~panel.apply(is_financial, axis=1)].copy()
    analytic = analytic[analytic["Z2_score"].notna()]

    nxt1 = analytic[["kode", "tahun", "laba_bersih", "total_ekuitas"]].copy()
    nxt1["tahun"] -= 1
    nxt1 = nxt1.rename(columns={"laba_bersih": "ni1", "total_ekuitas": "eq1"})
    df = analytic.merge(nxt1, on=["kode", "tahun"], how="inner")

    if definition == "persistent":
        nxt2 = analytic[["kode", "tahun", "laba_bersih"]].copy()
        nxt2["tahun"] -= 2
        nxt2 = nxt2.rename(columns={"laba_bersih": "ni2"})
        df = df.merge(nxt2, on=["kode", "tahun"], how="inner")
        df["y"] = (((df.ni1 < 0) & (df.ni2 < 0)) | (df.eq1 < 0)).astype(int)
    else:  # oneyear
        df["y"] = ((df.ni1 < 0) | (df.eq1 < 0)).astype(int)

    df = df.dropna(subset=FEATURES + ["Z2_score", "y"])
    return df


# --------------------------------------------------------------------------- #
# 3. Fit the three candidate models with grouped CV + within-fold SMOTE
# --------------------------------------------------------------------------- #
def fit_models(Xtr, ytr, groups):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    models = {}
    models["Logistic Regression"] = GridSearchCV(
        ImbPipeline([("sc", StandardScaler()), ("sm", SMOTE(random_state=SEED)),
                     ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"m__C": [0.01, 0.1, 1, 10]},
        cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr, groups=groups)
    models["Random Forest"] = GridSearchCV(
        ImbPipeline([("sm", SMOTE(random_state=SEED)),
                     ("m", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
        {"m__n_estimators": [300, 500], "m__max_depth": [None, 10, 20],
         "m__min_samples_leaf": [1, 5]},
        cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr, groups=groups)
    models["XGBoost"] = GridSearchCV(
        ImbPipeline([("sm", SMOTE(random_state=SEED)),
                     ("m", XGBClassifier(random_state=SEED, eval_metric="logloss",
                                         n_jobs=-1))]),
        {"m__n_estimators": [300, 500], "m__max_depth": [3, 5, 7],
         "m__learning_rate": [0.05, 0.1]},
        cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr, groups=groups)
    return models


def metrics_row(name, y, score, pred):
    return dict(model=name, AUC=roc_auc_score(y, score), Recall=recall_score(y, pred),
                Precision=precision_score(y, pred), F1=f1_score(y, pred),
                BalAcc=balanced_accuracy_score(y, pred))


# --------------------------------------------------------------------------- #
# 4. Primary analysis (one-year label; train t<=2023, test t=2024 -> FY2025)
# --------------------------------------------------------------------------- #
def run(panel, definition, train_max, test_year, tag):
    df = build_panel(panel, definition)
    tr = df[df.tahun <= train_max].copy()
    te = df[df.tahun == test_year].copy()

    lo, hi = tr[FEATURES].quantile(0.01), tr[FEATURES].quantile(0.99)  # winsorise once
    Xtr, Xte = tr[FEATURES].clip(lo, hi, axis=1), te[FEATURES].clip(lo, hi, axis=1)
    ytr, yte = tr.y.values, te.y.values
    z = te["Z2_score"].values

    print(f"\n[{tag}] train n={len(tr)} ({ytr.mean()*100:.1f}% distress) | "
          f"test n={len(te)} ({yte.mean()*100:.1f}%, {int(yte.sum())} distressed)")

    models = fit_models(Xtr, ytr, tr.kode.values)
    prob = {n: g.predict_proba(Xte)[:, 1] for n, g in models.items()}
    for n, g in models.items():
        print(f"   {n}: CV ROC-AUC={g.best_score_:.4f}  best={g.best_params_}")

    rows = [metrics_row("Altman Z\"-Score (cutoff 1.1)", yte, -z, (z < 1.1).astype(int))]
    for n, p in prob.items():
        rows.append(metrics_row(n, yte, p, (p >= 0.5).astype(int)))
    res = pd.DataFrame(rows).set_index("model").round(4)
    res.to_csv(os.path.join(RES, f"results_{tag}.csv"))
    print(res.to_string())

    return dict(models=models, Xte=Xte, yte=yte, z=z, prob=prob,
                test_kode=te.kode.values, res=res)


# --------------------------------------------------------------------------- #
# 5. Bootstrap CIs, calibration, SHAP, and figures (primary run only)
# --------------------------------------------------------------------------- #
def bootstrap_ci(yte, scores, B=2000):
    rng = np.random.default_rng(SEED)
    n = len(yte)
    boot = {k: [] for k in scores}
    diff = {k: [] for k in scores if not k.startswith("Altman")}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            continue
        a = {k: roc_auc_score(yte[idx], s[idx]) for k, s in scores.items()}
        for k in boot:
            boot[k].append(a[k])
        base = [k for k in scores if k.startswith("Altman")][0]
        for k in diff:
            diff[k].append(a[k] - a[base])
    rows = []
    for k, v in boot.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append(dict(quantity=f"AUC: {k}", lo=round(lo, 4), hi=round(hi, 4)))
    for k, v in diff.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append(dict(quantity=f"AUC diff ({k} - Altman)", lo=round(lo, 4), hi=round(hi, 4)))
    return pd.DataFrame(rows)


def make_figures(run_obj):
    Xte, yte, z, prob = run_obj["Xte"], run_obj["yte"], run_obj["z"], run_obj["prob"]
    res = run_obj["res"]
    rf = run_obj["models"]["Random Forest"].best_estimator_.named_steps["m"]
    COL = {"Altman Z\"-Score (cutoff 1.1)": "#8d99ae", "Logistic Regression": "#2a9d8f",
           "Random Forest": "#7b2cbf", "XGBoost": "#e76f51"}
    scores = {"Altman Z\"-Score (cutoff 1.1)": -z, **prob}

    # --- ROC + metric bars ---
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=150)
    for n, s in scores.items():
        fpr, tpr, _ = roc_curve(yte, s)
        ls = "--" if n.startswith("Altman") else "-"
        lbl = "Altman Z″-Score" if n.startswith("Altman") else n
        ax[0].plot(fpr, tpr, color=COL[n], ls=ls, lw=2.2,
                   label=f"{lbl} (AUC = {roc_auc_score(yte, s):.3f})")
    ax[0].plot([0, 1], [0, 1], color="#ccc", lw=1)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) ROC curves on the hold-out test set", fontsize=11)
    ax[0].legend(loc="lower right", fontsize=8.5, frameon=False); ax[0].grid(alpha=.22)
    ax[0].spines[["top", "right"]].set_visible(False)
    metrics = ["AUC", "Recall", "F1", "BalAcc"]
    labs = ["ROC-AUC", "Recall", "F1-score", "Balanced\naccuracy"]
    x = np.arange(len(metrics)); w = 0.19
    for i, n in enumerate(scores):
        vals = [res.loc[n, m] for m in metrics]
        lbl = "Altman Z″-Score" if n.startswith("Altman") else n
        b = ax[1].bar(x + (i - 1.5) * w, vals, w, color=COL[n], label=lbl,
                      edgecolor="white", linewidth=.6)
        for bb, v in zip(b, vals):
            ax[1].text(bb.get_x() + bb.get_width() / 2, v + .012, f"{v:.2f}",
                       ha="center", va="bottom", fontsize=7.3, color="#333")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, fontsize=9.5); ax[1].set_ylim(0, 1.09)
    ax[1].set_ylabel("Score"); ax[1].set_title("(b) Predictive performance by metric", fontsize=11)
    ax[1].legend(fontsize=8.5, frameon=False, loc="upper right"); ax[1].grid(axis="y", alpha=.22)
    ax[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_roc_metrics.png"), dpi=300,
                                    bbox_inches="tight"); plt.close()

    # --- calibration ---
    p_rf = prob["Random Forest"]
    frac, mp = calibration_curve(yte, p_rf, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.6, 5.2), dpi=150)
    plt.plot([0, 1], [0, 1], color="#ccc", lw=1, label="Perfect calibration")
    plt.plot(mp, frac, "o-", color="#7b2cbf", lw=2,
             label=f"Random Forest (Brier = {brier_score_loss(yte, p_rf):.3f})")
    plt.xlabel("Mean model output for the distressed class")
    plt.ylabel("Observed distress frequency")
    plt.title("Calibration of the selected Random Forest model", fontsize=10.5)
    plt.legend(frameon=False, fontsize=9); plt.grid(alpha=.22)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_calibration.png"), dpi=300,
                                    bbox_inches="tight"); plt.close()

    # --- SHAP global + local ---
    expl = shap.TreeExplainer(rf)
    sv = expl.shap_values(Xte)
    sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    mean_abs = np.abs(sv1).mean(axis=0)
    rank = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs}) \
        .sort_values("mean_abs_shap", ascending=False)
    rank.to_csv(os.path.join(RES, "shap_ranking.csv"), index=False)
    print("\nGlobal SHAP ranking:\n", rank.to_string(index=False))

    LBL = {"X1": "X₁ (WC/TA)", "X2": "X₂ (RE/TA)", "X3": "X₃ (EBIT/TA)",
           "X4": "X₄ (BVE/TL)", "ROA": "ROA", "leverage": "Leverage (TL/TA)",
           "current_ratio": "Current ratio"}
    fig = plt.figure(figsize=(13, 5.6), dpi=150)
    fig.add_subplot(1, 2, 1)
    shap.summary_plot(sv1, Xte.rename(columns=LBL), show=False, plot_size=None, max_display=7)
    plt.gca().set_title("(a) SHAP summary (test set)", fontsize=11)
    ax2 = fig.add_subplot(1, 2, 2)
    o = np.argsort(mean_abs); zc = {"X1", "X2", "X3", "X4"}
    cols = ["#7b2cbf" if FEATURES[i] in zc else "#2a9d8f" for i in o]
    ax2.barh([LBL[FEATURES[i]] for i in o], mean_abs[o], color=cols,
             edgecolor="white", height=.62)
    ax2.set_xlabel("Mean |SHAP value|"); ax2.set_title("(b) Global feature importance", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False); ax2.grid(axis="x", alpha=.22)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_shap.png"), dpi=300,
                                    bbox_inches="tight"); plt.close()

    # local: true positive closest to median model output
    tp = np.where((yte == 1) & (p_rf >= 0.5))[0]
    med = np.median(p_rf[tp]); pick = tp[np.argmin(np.abs(p_rf[tp] - med))]
    ev = expl.expected_value
    base = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
    exp = shap.Explanation(values=sv1[pick], base_values=base,
                           data=Xte.iloc[pick].values,
                           feature_names=[LBL[f] for f in FEATURES])
    plt.figure(figsize=(7.5, 4.6), dpi=150)
    shap.plots.waterfall(exp, show=False, max_display=7)
    plt.title(f"Local explanation - median true positive (output = {p_rf[pick]:.2f})",
              fontsize=9.5)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_shap_local.png"), dpi=300,
                                    bbox_inches="tight"); plt.close()
    print(f"\nLocal SHAP case: {run_obj['test_kode'][pick]} (output {p_rf[pick]:.3f})")


# --------------------------------------------------------------------------- #
def main():
    panel = load_panel()
    print(f"Merged firm-year records: {len(panel)}")

    # Primary analysis: one-year label, test FY2024 -> FY2025
    primary = run(panel, "oneyear", train_max=2023, test_year=2024, tag="primary")

    # Bootstrap CIs
    scores = {"Altman Z\"-Score (cutoff 1.1)": -primary["z"], **primary["prob"]}
    ci = bootstrap_ci(primary["yte"], scores)
    ci.to_csv(os.path.join(RES, "bootstrap_ci.csv"), index=False)
    print("\nBootstrap 95% CIs:\n", ci.to_string(index=False))

    # Figures + SHAP
    make_figures(primary)

    # Sensitivity analysis: persistent label, test FY2023 -> FY2024-FY2025
    run(panel, "persistent", train_max=2022, test_year=2023, tag="sensitivity")

    print("\nDone. See ../results/ and ../figures/.")


if __name__ == "__main__":
    main()
