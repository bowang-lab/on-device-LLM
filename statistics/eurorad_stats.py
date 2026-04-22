#!/usr/bin/env python3
"""
Eurorad statistical analysis (nominal diagnostic accuracy task).

Computes:
  1. Majority-vote accuracy with Wilson Score 95% CIs
  2. Per-subspecialty accuracy + Fisher's exact test
  3. Pairwise McNemar's test with Holm-Bonferroni correction + odds ratios
  4. Fleiss' kappa (intra-model stability) with bootstrap 95% CIs
  5. Cohen's kappa (inter-model agreement) with bootstrap 95% CIs

Usage:
    python statistics/eurorad_stats.py
    python statistics/eurorad_stats.py --output statistics/eurorad_results.csv
"""

import csv
import argparse
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score

from utils import (
    norm_text, wilson_ci, fmt_pct, fmt_p, fmt_kappa,
    mcnemar_test, holm_bonferroni, fleiss_kappa,
    bootstrap_ci, print_section,
)


# ─── Model definitions ───────────────────────────────────────────────────────
# Excluded: GPT-5 (0807), GPT-5.2 (1211), o4-mini

MODELS = {
    "GPT-5.1":          ["gpt-5.1-1113-M1", "gpt-5.1-1113-M2", "gpt-5.1-1113-M3"],
    "GPT-5-mini":       ["gpt-5-mini-0807-M1", "gpt-5-mini-0807-M2", "gpt-5-mini-0807-M3"],
    "Gemini 3.1 Pro":   ["gemini-3.1-pro-M1", "gemini-3.1-pro-M2", "gemini-3.1-pro-M3"],
    "DeepSeek-R1":      ["deepseek r1 0528 v2", "deepseek r1 0528 v2.1", "deepseek r1 0528 v2.2"],
    "gpt-oss-20b (H)":  ["oss-20b (H) v1", "oss-20b (H) v2", "oss-20b (H) v3"],
    "gpt-oss-20b (13B)": ["oss20b-13beams-v1", "oss20b-13beams-v2", "oss20b-13beams-v3"],
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


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_eurorad():
    with open("csvs/final_csvs/Eurorad.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_headers = next(reader)
        rows = list(reader)

    seen = Counter()
    headers = []
    for h in raw_headers:
        count = seen[h]
        seen[h] += 1
        headers.append(h if count == 0 else f"{h}.{count}")

    df = pd.DataFrame(rows, columns=headers)
    df = df[df["case_id"].str.strip().astype(bool)].copy()
    df["Section"] = df["Section"].apply(lambda s: "Others" if s in OTHERS_SECTIONS else s)
    return df


# ─── Majority vote ────────────────────────────────────────────────────────────

def compute_majority_vote(df):
    gt = df["FinalDiagnosis"].apply(norm_text).values
    results = {}
    per_run = {}

    for model_name, cols in MODELS.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  WARNING: {model_name} missing columns: {missing}")
            continue

        run_correct = []
        for col in cols:
            preds = df[col].apply(norm_text).values
            run_correct.append([p == g for p, g in zip(preds, gt)])

        n_cases = len(gt)
        majority = [sum(run_correct[r][i] for r in range(len(cols))) >= 2
                     for i in range(n_cases)]
        results[model_name] = majority
        per_run[model_name] = run_correct

    return results, per_run


# ─── 1. Overall accuracy ─────────────────────────────────────────────────────

def report_accuracy(majority):
    print_section("1. Overall Accuracy (Majority Vote, Wilson 95% CI)")
    print(f"  {'Model':<22s}  {'Acc':>6s}  {'95% CI':>20s}  {'k/n':>8s}")
    print(f"  {'-'*60}")
    for m in MODELS:
        if m not in majority:
            continue
        correct = majority[m]
        k, n = sum(correct), len(correct)
        p, lo, hi = wilson_ci(k, n)
        print(f"  {m:<22s}  {p*100:5.1f}%  ({lo*100:5.1f} -- {hi*100:5.1f})  {k:3d}/{n}")


# ─── 2. Per-subspecialty accuracy + Fisher's exact ────────────────────────────

def report_subspecialty(df, majority):
    print_section("2. Per-Subspecialty Accuracy")
    sections = df["Section"].values
    model_names = [m for m in MODELS if m in majority]

    header = f"  {'Subspecialty':<30s}"
    for m in model_names:
        header += f"  {m:>10s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for cat in CATEGORIES:
        mask = [s == cat for s in sections]
        n_cat = sum(mask)
        row = f"  {cat:<30s}"
        for m in model_names:
            k_cat = sum(c for c, msk in zip(majority[m], mask) if msk)
            p = k_cat / n_cat * 100 if n_cat > 0 else 0
            row += f"  {p:>9.1f}%"
        print(row)

    # Fisher's exact: compare best on-device vs best proprietary per subspecialty
    print_section("2b. Per-Subspecialty Fisher's Exact Test (best on-device vs best proprietary)")
    proprietary = ["GPT-5.1", "GPT-5-mini", "Gemini 3.1 Pro", "DeepSeek-R1"]
    on_device = ["gpt-oss-120b (H)", "Qwen3.5 35B", "Qwen3.5 35B FT"]

    p_values = []
    for cat in CATEGORIES:
        mask = np.array([s == cat for s in sections])
        n_cat = mask.sum()
        if n_cat == 0:
            continue

        # Best proprietary accuracy
        best_prop_k = max(sum(c for c, m in zip(majority[m_name], mask) if m)
                         for m_name in proprietary if m_name in majority)
        # Best on-device accuracy
        best_dev_k = max(sum(c for c, m in zip(majority[m_name], mask) if m)
                        for m_name in on_device if m_name in majority)

        # 2x2 table for Fisher's exact
        table = np.array([
            [best_prop_k, n_cat - best_prop_k],
            [best_dev_k, n_cat - best_dev_k],
        ])
        _, p_val = fisher_exact(table)
        p_values.append((cat, p_val))

    corrected = holm_bonferroni(p_values)
    print(f"  {'Subspecialty':<35s}  {'p (raw)':>10s}  {'p (adj)':>10s}  {'Sig':>5s}")
    print(f"  {'-'*65}")
    for label, raw_p, adj_p, sig in corrected:
        sig_str = "*" if sig else ""
        print(f"  {label:<35s}  {fmt_p(raw_p):>10s}  {fmt_p(adj_p):>10s}  {sig_str:>5s}")


# ─── 3. McNemar's with Holm-Bonferroni + odds ratios ─────────────────────────

def report_mcnemar(majority):
    print_section("3. Pairwise McNemar's Test (Holm-Bonferroni corrected)")
    model_names = [m for m in MODELS if m in majority]
    pairs = list(combinations(model_names, 2))

    raw_results = []
    for m1, m2 in pairs:
        p_val, odds = mcnemar_test(majority[m1], majority[m2])
        raw_results.append(((m1, m2), p_val, odds))

    # Apply Holm-Bonferroni
    p_list = [(f"{m1} vs {m2}", p) for (m1, m2), p, _ in raw_results]
    corrected = holm_bonferroni(p_list)

    print(f"  {'Comparison':<45s}  {'p (raw)':>10s}  {'p (adj)':>10s}  {'OR':>8s}  {'Sig':>5s}")
    print(f"  {'-'*85}")

    for i, ((m1, m2), raw_p, odds) in enumerate(raw_results):
        _, _, adj_p, sig = corrected[i]
        sig_str = "*" if sig else ""
        odds_str = f"{odds:.2f}" if odds != float("inf") else "inf"
        print(f"  {m1 + ' vs ' + m2:<45s}  {fmt_p(raw_p):>10s}  {fmt_p(adj_p):>10s}  {odds_str:>8s}  {sig_str:>5s}")

    n_sig = sum(1 for _, _, _, sig in corrected if sig)
    print(f"\n  {n_sig}/{len(corrected)} comparisons significant after correction (α=0.05)")


# ─── 4. Fleiss' kappa with bootstrap CIs ─────────────────────────────────────

def report_fleiss(per_run):
    print_section("4. Fleiss' Kappa (Intra-Model Stability, 3 runs, bootstrap 95% CI)")

    for model_name in MODELS:
        if model_name not in per_run:
            continue
        runs = per_run[model_name]
        n_cases = len(runs[0])
        n_raters = len(runs)

        # Build table
        table = np.zeros((n_cases, 2), dtype=int)
        for r in range(n_raters):
            for i in range(n_cases):
                table[i, 1 if runs[r][i] else 0] += 1

        def kappa_fn(tbl):
            return fleiss_kappa(tbl)

        point, lo, hi = bootstrap_ci(table, kappa_fn, n_boot=2000)
        print(f"  {model_name:<22s}: κ = {fmt_kappa(point, lo, hi)}")


# ─── 5. Cohen's kappa with bootstrap CIs ─────────────────────────────────────

def report_cohen(majority):
    print_section("5. Cohen's Kappa (Inter-Model Agreement, bootstrap 95% CI)")
    model_names = [m for m in MODELS if m in majority]

    for m1, m2 in combinations(model_names, 2):
        a = np.array([int(c) for c in majority[m1]])
        b = np.array([int(c) for c in majority[m2]])

        def kappa_fn(data):
            return cohen_kappa_score(data[0], data[1])

        point, lo, hi = bootstrap_ci((a, b), kappa_fn, n_boot=2000)
        print(f"  {m1:>22s} vs {m2:<22s}: κ = {fmt_kappa(point, lo, hi)}")


# ─── Save CSV ─────────────────────────────────────────────────────────────────

def save_results(df, majority, output):
    sections = df["Section"].values
    rows = []

    for m in MODELS:
        if m not in majority:
            continue
        correct = majority[m]
        k, n = sum(correct), len(correct)
        p, lo, hi = wilson_ci(k, n)
        row = {"model": m, "category": "Overall",
               "accuracy": round(p * 100, 1),
               "ci_lo": round(lo * 100, 1), "ci_hi": round(hi * 100, 1),
               "k": k, "n": n}
        rows.append(row)

        for cat in CATEGORIES:
            mask = [s == cat for s in sections]
            n_cat = sum(mask)
            k_cat = sum(c for c, msk in zip(correct, mask) if msk)
            p, lo, hi = wilson_ci(k_cat, n_cat)
            rows.append({"model": m, "category": cat,
                         "accuracy": round(p * 100, 1),
                         "ci_lo": round(lo * 100, 1), "ci_hi": round(hi * 100, 1),
                         "k": k_cat, "n": n_cat})

    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nSaved: {output}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", default="statistics/eurorad_results.csv")
    args = ap.parse_args()

    print("Loading Eurorad data...")
    df = load_eurorad()
    df = df[df["FinalDiagnosis"].apply(lambda x: bool(str(x).strip()))].copy()
    print(f"  {len(df)} cases with ground truth")

    majority, per_run = compute_majority_vote(df)

    report_accuracy(majority)
    report_subspecialty(df, majority)
    report_mcnemar(majority)
    report_fleiss(per_run)
    report_cohen(majority)
    save_results(df, majority, args.output)


if __name__ == "__main__":
    main()
