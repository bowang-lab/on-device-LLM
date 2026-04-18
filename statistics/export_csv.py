#!/usr/bin/env python3
"""
Export all statistics to a single consolidated CSV.

Usage:
    python statistics/export_csv.py
    python statistics/export_csv.py --output statistics/all_statistics.csv
"""

import csv
import argparse
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon
from sklearn.metrics import cohen_kappa_score

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    norm_text, wilson_ci, mcnemar_test, holm_bonferroni,
    fleiss_kappa, bootstrap_ci,
)


# ─── Eurorad definitions ─────────────────────────────────────────────────────

EURORAD_MODELS = {
    "GPT-5.1":          ["gpt-5.1-1113-M1", "gpt-5.1-1113-M2", "gpt-5.1-1113-M3"],
    "GPT-5-mini":       ["gpt-5-mini-0807-M1", "gpt-5-mini-0807-M2", "gpt-5-mini-0807-M3"],
    "Gemini 3.1 Pro":   ["gemini-3.1-pro-M1", "gemini-3.1-pro-M2", "gemini-3.1-pro-M3"],
    "DeepSeek-R1":      ["deepseek r1 0528 v2", "deepseek r1 0528 v2.1", "deepseek r1 0528 v2.2"],
    "gpt-oss-20b (H)":  ["oss-20b (H) v1", "oss-20b (H) v2", "oss-20b (H) v3"],
    "gpt-oss-120b (H)": ["oss-120b (H) v1", "oss-120b (H) v2", "oss-120b (H) v3"],
    "Qwen3.5 9B":       ["qwen3.5 9B v1", "qwen3.5 9B v2", "qwen3.5 9B v3"],
    "Qwen3.5 27B":      ["qwen3.5 27B v1", "qwen3.5 27B v2", "qwen3.5 27B v3"],
    "Qwen3.5 35B":      ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
    "Qwen3.5 35B FT":   ["qwen3.5 35B fine-tuned v1", "qwen3.5 35B fine-tuned v2", "qwen3.5 35B fine-tuned v3"],
    "Gemma 4 31B":       ["gemma-4-31b-M1", "gemma-4-31b-M2", "gemma-4-31b-M3"],
}

CATEGORIES = [
    "Musculoskeletal system", "Paediatric radiology", "Cardiovascular",
    "Abdominal imaging", "Uroradiology & genital male imaging", "Others",
    "Breast imaging", "Head & neck imaging", "Neuroradiology", "Chest imaging",
]
OTHERS_SECTIONS = {"Genital (female) imaging", "Interventional radiology"}

# ─── NMED definitions ─────────────────────────────────────────────────────────

NMED_MODELS = {
    "GPT-5.1": {
        "diag": ["gpt-5.1-1113-m1", "gpt-5.1-1113-m2", "gpt-5.1-1113-m3"],
        "treat": ["gpt-5.1-1113-m1", "gpt-5.1-1113-m2", "gpt-5.1-1113-m3"],
    },
    "GPT-5-mini": {
        "diag": ["gpt-5-mini-0807-m1", "gpt-5-mini-0807-m2", "gpt-5-mini-0807-m3"],
        "treat": ["gpt-5-mini-0807-m1", "gpt-5-mini-0807-m2", "gpt-5-mini-0807-m3"],
    },
    "Gemini 3.1": {
        "diag": ["gemini-3.1-pro-m1", "gemini-3.1-pro-m2", "gemini-3.1-pro-m3"],
        "treat": ["gemini-3.1-pro-m1", "gemini-3.1-pro-m2", "gemini-3.1-pro-m3"],
    },
    "DeepSeek-R1": {
        "diag": ["deepseek-0528-v1", "deepseek-0528-v2", "deepseek-0528-v3"],
        "treat": ["deepseek-r1-0528-v1", "deepseek-r1-0528-v2", "deepseek-r1-0528-v3"],
    },
    "gpt-oss-20b (H)": {
        "diag": ["oss20b (H) v1", "oss20b (H) v2", "oss20b (H) v3"],
        "treat": ["oss20b (H) v1", "oss20b (H) v2", "oss20b (H) v3"],
    },
    "gpt-oss-120b (H)": {
        "diag": ["oss120b (H) v1", "oss120b (H) v2", "oss120b (H) v3"],
        "treat": ["oss120b (H) v1", "oss120b (H) v2", "oss120b (H) v3"],
    },
    "Qwen3.5 9B": {
        "diag": ["qwen3.5 9B v1", "qwen3.5 9B v2", "qwen3.5 9B v3"],
        "treat": ["qwen3.5 9B v1", "qwen3.5 9B v2", "qwen3.5 9B v3"],
    },
    "Qwen3.5 27B": {
        "diag": ["qwen3.5 27B v1", "qwen3.5 27B v2", "qwen3.5 27B v3"],
        "treat": ["qwen3.5 27B v1", "qwen3.5 27B v2", "qwen3.5 27B v3"],
    },
    "Qwen3.5 35B": {
        "diag": ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
        "treat": ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
    },
    "Gemma 4 31B": {
        "diag": ["gemma4 31B v1", "gemma4 31B v2", "gemma4 31B v3"],
        "treat": ["gemma4 31B v1", "gemma4 31B v2", "gemma4 31B v3"],
    },
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_eurorad():
    with open("csvs/final_csvs/Eurorad.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_headers = next(reader)
        rows_data = list(reader)
    seen = Counter()
    headers = []
    for h in raw_headers:
        count = seen[h]; seen[h] += 1
        headers.append(h if count == 0 else f"{h}.{count}")
    df = pd.DataFrame(rows_data, columns=headers)
    df = df[df["case_id"].str.strip().astype(bool)].copy()
    df["Section"] = df["Section"].apply(lambda s: "Others" if s in OTHERS_SECTIONS else s)
    df = df[df["FinalDiagnosis"].apply(lambda x: bool(str(x).strip()))].copy()
    return df


def load_nmed():
    df_diag = pd.read_csv("csvs/final_csvs/NMED_Diagnosis.csv")
    df_treat = pd.read_csv("csvs/final_csvs/NMED_Treatment.csv")
    df_diag = df_diag[df_diag["Clinical specialty"].notna()]
    df_treat = df_treat[df_treat["Clinical specialty"].notna()]
    return df_diag, df_treat


def eurorad_majority_vote(df):
    gt = df["FinalDiagnosis"].apply(norm_text).values
    results, per_run = {}, {}
    for model_name, cols in EURORAD_MODELS.items():
        if any(c not in df.columns for c in cols):
            continue
        run_correct = []
        for col in cols:
            preds = df[col].apply(norm_text).values
            run_correct.append([p == g for p, g in zip(preds, gt)])
        n = len(gt)
        majority = [sum(run_correct[r][i] for r in range(len(cols))) >= 2
                     for i in range(n)]
        results[model_name] = majority
        per_run[model_name] = run_correct
    return results, per_run


def nmed_consensus(df, model_cols):
    gt = pd.to_numeric(df["HumanEvalScore"], errors="coerce")
    scores = [pd.to_numeric(df[c], errors="coerce") for c in model_cols]
    consensus = pd.concat(scores, axis=1).mean(axis=1)
    mask = gt.notna() & consensus.notna()
    return gt[mask].values, consensus[mask].values


def nmed_per_run(df, model_cols):
    gt = pd.to_numeric(df["HumanEvalScore"], errors="coerce")
    runs = [pd.to_numeric(df[c], errors="coerce") for c in model_cols]
    mask = gt.notna()
    for s in runs:
        mask = mask & s.notna()
    return gt[mask].values, [s[mask].values for s in runs]


def icc_3k(ratings):
    n, k = ratings.shape
    gm = ratings.mean()
    ss_r = k * np.sum((ratings.mean(axis=1) - gm) ** 2)
    ss_c = n * np.sum((ratings.mean(axis=0) - gm) ** 2)
    ss_t = np.sum((ratings - gm) ** 2)
    ss_e = ss_t - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / ms_r if ms_r > 0 else 0.0


def rank_biserial(e1, e2):
    d = np.abs(e1) - np.abs(e2)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    rp = ranks[d > 0].sum()
    rm = ranks[d < 0].sum()
    return (rp - rm) / (rp + rm)


# ─── Row builder ──────────────────────────────────────────────────────────────

def R(dataset, analysis, metric, model_1="", model_2="", category="",
      value=None, ci_lo=None, ci_hi=None, p_raw=None, p_adj=None,
      significant=None, effect_size=None, n=None, k=None):
    return {
        "dataset": dataset, "analysis": analysis, "metric": metric,
        "model_1": model_1, "model_2": model_2, "category": category,
        "value": _r(value), "ci_lo": _r(ci_lo), "ci_hi": _r(ci_hi),
        "p_raw": _r(p_raw), "p_adj": _r(p_adj),
        "significant": significant if significant is not None else "",
        "effect_size": _r(effect_size), "n": n if n is not None else "",
        "k": k if k is not None else "",
    }

def _r(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 4)
    return v


# ─── Eurorad computations ────────────────────────────────────────────────────

def eurorad_rows(df, majority, per_run):
    rows = []
    sections = df["Section"].values

    # 1. Accuracy (overall + per subspecialty)
    for m in EURORAD_MODELS:
        if m not in majority:
            continue
        correct = majority[m]
        k_val, n_val = sum(correct), len(correct)
        p, lo, hi = wilson_ci(k_val, n_val)
        rows.append(R("Eurorad", "accuracy", "accuracy_pct",
                       model_1=m, category="Overall",
                       value=p*100, ci_lo=lo*100, ci_hi=hi*100,
                       n=n_val, k=k_val))
        for cat in CATEGORIES:
            mask = [s == cat for s in sections]
            n_cat = sum(mask)
            k_cat = sum(c for c, msk in zip(correct, mask) if msk)
            p, lo, hi = wilson_ci(k_cat, n_cat)
            rows.append(R("Eurorad", "accuracy", "accuracy_pct",
                           model_1=m, category=cat,
                           value=p*100, ci_lo=lo*100, ci_hi=hi*100,
                           n=n_cat, k=k_cat))

    # 2. Per-subspecialty Fisher's exact (best on-device vs best proprietary)
    proprietary = ["GPT-5.1", "GPT-5-mini", "Gemini 3.1 Pro", "DeepSeek-R1"]
    on_device = ["gpt-oss-120b (H)", "Qwen3.5 35B", "Qwen3.5 35B FT"]
    fisher_pvals = []
    for cat in CATEGORIES:
        mask = np.array([s == cat for s in sections])
        n_cat = mask.sum()
        if n_cat == 0:
            continue
        best_prop = max(sum(c for c, m in zip(majority[mn], mask) if m)
                        for mn in proprietary if mn in majority)
        best_dev = max(sum(c for c, m in zip(majority[mn], mask) if m)
                       for mn in on_device if mn in majority)
        table = np.array([[best_prop, n_cat - best_prop],
                          [best_dev, n_cat - best_dev]])
        _, pv = fisher_exact(table)
        fisher_pvals.append((cat, pv))

    corrected = holm_bonferroni(fisher_pvals)
    for label, raw_p, adj_p, sig in corrected:
        rows.append(R("Eurorad", "fisher_exact_subspecialty", "p_value",
                       model_1="best_proprietary", model_2="best_on_device",
                       category=label, p_raw=raw_p, p_adj=adj_p,
                       significant="yes" if sig else "no"))

    # 3. McNemar's pairwise with Holm-Bonferroni
    model_names = [m for m in EURORAD_MODELS if m in majority]
    pairs = list(combinations(model_names, 2))
    mcn_raw = []
    for m1, m2 in pairs:
        pv, odds = mcnemar_test(majority[m1], majority[m2])
        mcn_raw.append(((m1, m2), pv, odds))
    p_list = [(f"{m1} vs {m2}", p) for (m1, m2), p, _ in mcn_raw]
    corrected = holm_bonferroni(p_list)
    for i, ((m1, m2), raw_p, odds) in enumerate(mcn_raw):
        _, _, adj_p, sig = corrected[i]
        rows.append(R("Eurorad", "mcnemar", "p_value",
                       model_1=m1, model_2=m2,
                       p_raw=raw_p, p_adj=adj_p,
                       effect_size=odds if odds != float("inf") else None,
                       significant="yes" if sig else "no"))

    # 4. Fleiss' kappa with bootstrap CIs
    for m in EURORAD_MODELS:
        if m not in per_run:
            continue
        runs = per_run[m]
        n_cases = len(runs[0])
        table = np.zeros((n_cases, 2), dtype=int)
        for r in range(len(runs)):
            for i_case in range(n_cases):
                table[i_case, 1 if runs[r][i_case] else 0] += 1
        pt, lo, hi = bootstrap_ci(table, fleiss_kappa, n_boot=2000)
        rows.append(R("Eurorad", "fleiss_kappa", "kappa",
                       model_1=m, value=pt, ci_lo=lo, ci_hi=hi))

    # 5. Cohen's kappa with bootstrap CIs
    for m1, m2 in combinations(model_names, 2):
        a = np.array([int(c) for c in majority[m1]])
        b = np.array([int(c) for c in majority[m2]])
        def kfn(data):
            return cohen_kappa_score(data[0], data[1])
        pt, lo, hi = bootstrap_ci((a, b), kfn, n_boot=2000)
        rows.append(R("Eurorad", "cohen_kappa", "kappa",
                       model_1=m1, model_2=m2,
                       value=pt, ci_lo=lo, ci_hi=hi))

    return rows


# ─── NMED computations ───────────────────────────────────────────────────────

def nmed_rows(df_diag, df_treat):
    rows = []

    for task, df, key in [("NMED_Diagnosis", df_diag, "diag"),
                           ("NMED_Treatment", df_treat, "treat")]:

        # 1. Median error + IQR
        model_errors = {}
        for m, cols in NMED_MODELS.items():
            gt, cons = nmed_consensus(df, cols[key])
            errors = cons - gt
            model_errors[m] = errors
            med = np.median(errors)
            q1, q3 = np.percentile(errors, [25, 75])
            rows.append(R(task, "error_summary", "median_error",
                           model_1=m, value=med,
                           ci_lo=q1, ci_hi=q3, n=len(errors)))

        # 2. Wilcoxon pairwise with Holm-Bonferroni + rank-biserial
        model_names = list(NMED_MODELS.keys())
        pairs = list(combinations(model_names, 2))
        wilc_raw = []
        for m1, m2 in pairs:
            e1, e2 = model_errors[m1], model_errors[m2]
            n_min = min(len(e1), len(e2))
            ae1, ae2 = np.abs(e1[:n_min]), np.abs(e2[:n_min])
            diffs = ae1 - ae2
            nonzero = diffs[diffs != 0]
            if len(nonzero) < 10:
                wilc_raw.append(((m1, m2), 1.0, 0.0))
                continue
            _, pv = wilcoxon(ae1, ae2)
            rb = rank_biserial(e1[:n_min], e2[:n_min])
            wilc_raw.append(((m1, m2), pv, rb))

        p_list = [(f"{m1} vs {m2}", p) for (m1, m2), p, _ in wilc_raw]
        corrected = holm_bonferroni(p_list)
        for i, ((m1, m2), raw_p, rb) in enumerate(wilc_raw):
            _, _, adj_p, sig = corrected[i]
            rows.append(R(task, "wilcoxon", "p_value",
                           model_1=m1, model_2=m2,
                           p_raw=raw_p, p_adj=adj_p,
                           effect_size=rb,
                           significant="yes" if sig else "no"))

        # 3. ICC(3,k) with bootstrap CIs
        for m, cols in NMED_MODELS.items():
            gt, run_scores = nmed_per_run(df, cols[key])
            errors = np.column_stack([r - gt for r in run_scores])
            pt, lo, hi = bootstrap_ci(errors, icc_3k, n_boot=2000)
            rows.append(R(task, "icc_3k", "icc",
                           model_1=m, value=pt, ci_lo=lo, ci_hi=hi))

        # 4. Weighted kappa with bootstrap CIs
        model_cons_disc = {}
        for m, cols in NMED_MODELS.items():
            gt, cons = nmed_consensus(df, cols[key])
            disc = np.round(cons * 2) / 2
            model_cons_disc[m] = np.array([f"{v:.1f}" for v in disc])
        all_labels = sorted(set(
            lb for arr in model_cons_disc.values() for lb in arr
        ))
        for m1, m2 in combinations(model_names, 2):
            s1, s2 = model_cons_disc[m1], model_cons_disc[m2]
            n_min = min(len(s1), len(s2))
            def kfn(data, _lb=all_labels):
                return cohen_kappa_score(data[0], data[1],
                                         labels=_lb, weights="linear")
            pt, lo, hi = bootstrap_ci((s1[:n_min], s2[:n_min]), kfn, n_boot=2000)
            rows.append(R(task, "weighted_kappa", "kappa_w",
                           model_1=m1, model_2=m2,
                           value=pt, ci_lo=lo, ci_hi=hi))

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", default="statistics/all_statistics.csv")
    args = ap.parse_args()

    print("Loading data...")
    df_euro = load_eurorad()
    df_diag, df_treat = load_nmed()
    print(f"  Eurorad: {len(df_euro)} cases")
    print(f"  NMED Diagnosis: {len(df_diag)}, Treatment: {len(df_treat)}")

    print("Computing Eurorad statistics...")
    majority, per_run = eurorad_majority_vote(df_euro)
    euro_rows = eurorad_rows(df_euro, majority, per_run)
    print(f"  {len(euro_rows)} rows")

    print("Computing NMED statistics...")
    nmed = nmed_rows(df_diag, df_treat)
    print(f"  {len(nmed)} rows")

    all_rows = euro_rows + nmed
    columns = ["dataset", "analysis", "metric", "model_1", "model_2",
               "category", "value", "ci_lo", "ci_hi", "p_raw", "p_adj",
               "significant", "effect_size", "n", "k"]
    df_out = pd.DataFrame(all_rows, columns=columns)
    df_out.to_csv(args.output, index=False)
    print(f"\nSaved {len(df_out)} rows to {args.output}")


if __name__ == "__main__":
    main()
