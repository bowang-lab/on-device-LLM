#!/usr/bin/env python3
"""
NMED statistical analysis (ordinal clinical judgment tasks).

Computes:
  1. Median error with IQR per model (consensus = mean of 3 runs)
  2. Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction
     + rank-biserial effect sizes
  3. ICC(3,k) for intra-model stability with bootstrap 95% CIs
  4. Linear weighted kappa for inter-model agreement with bootstrap 95% CIs

Usage:
    python statistics/nmed_stats.py
    python statistics/nmed_stats.py --output statistics/nmed_results.csv
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score

from utils import (
    holm_bonferroni, bootstrap_ci, fmt_p, fmt_kappa, print_section,
)


# ─── Model definitions ───────────────────────────────────────────────────────
# Excluded: GPT-5, GPT-5.2, o4-mini

MODELS = {
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

def load_nmed():
    df_diag = pd.read_csv("csvs/final_csvs/NMED_Diagnosis.csv")
    df_treat = pd.read_csv("csvs/final_csvs/NMED_Treatment.csv")
    df_diag = df_diag[df_diag["Clinical specialty"].notna()]
    df_treat = df_treat[df_treat["Clinical specialty"].notna()]
    return df_diag, df_treat


def get_consensus_scores(df, model_cols):
    """Mean of 3 runs as consensus score per case."""
    gt = pd.to_numeric(df["HumanEvalScore"], errors="coerce")
    scores = []
    for col in model_cols:
        scores.append(pd.to_numeric(df[col], errors="coerce"))
    consensus = pd.concat(scores, axis=1).mean(axis=1)
    mask = gt.notna() & consensus.notna()
    return gt[mask].values, consensus[mask].values


def get_per_run_scores(df, model_cols):
    """Individual run scores aligned with GT."""
    gt = pd.to_numeric(df["HumanEvalScore"], errors="coerce")
    run_scores = []
    for col in model_cols:
        run_scores.append(pd.to_numeric(df[col], errors="coerce"))
    mask = gt.notna()
    for s in run_scores:
        mask = mask & s.notna()
    return gt[mask].values, [s[mask].values for s in run_scores]


# ─── 1. Median error + IQR ───────────────────────────────────────────────────

def report_error_summary(df_diag, df_treat):
    print_section("1. Median Error (consensus − GT) with IQR")
    for task_name, df, key in [("Diagnosis", df_diag, "diag"),
                                ("Treatment", df_treat, "treat")]:
        print(f"\n  --- {task_name} ---")
        print(f"  {'Model':<22s}  {'Median':>8s}  {'IQR':>20s}  {'N':>5s}")
        print(f"  {'-'*60}")
        for m, cols in MODELS.items():
            gt, consensus = get_consensus_scores(df, cols[key])
            errors = consensus - gt
            med = np.median(errors)
            q1, q3 = np.percentile(errors, [25, 75])
            print(f"  {m:<22s}  {med:>8.2f}  ({q1:>6.2f} -- {q3:>6.2f})  {len(errors):>5d}")


# ─── 2. Wilcoxon signed-rank + Holm-Bonferroni + effect size ──────────────────

def rank_biserial(errors_a, errors_b):
    """Rank-biserial correlation as effect size for Wilcoxon."""
    diffs = np.abs(errors_a) - np.abs(errors_b)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    n = len(diffs)
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1
    R_plus = ranks[diffs > 0].sum()
    R_minus = ranks[diffs < 0].sum()
    return (R_plus - R_minus) / (R_plus + R_minus)


def report_wilcoxon(df_diag, df_treat):
    print_section("2. Pairwise Wilcoxon Signed-Rank (Holm-Bonferroni corrected)")

    for task_name, df, key in [("Diagnosis", df_diag, "diag"),
                                ("Treatment", df_treat, "treat")]:
        print(f"\n  --- {task_name} ---")

        # Compute consensus errors per model
        model_errors = {}
        for m, cols in MODELS.items():
            gt, consensus = get_consensus_scores(df, cols[key])
            model_errors[m] = consensus - gt

        model_names = list(MODELS.keys())
        pairs = list(combinations(model_names, 2))

        raw_results = []
        for m1, m2 in pairs:
            e1, e2 = model_errors[m1], model_errors[m2]
            # Align on min length (should be same but safety)
            n = min(len(e1), len(e2))
            abs_e1, abs_e2 = np.abs(e1[:n]), np.abs(e2[:n])
            diffs = abs_e1 - abs_e2
            nonzero = diffs[diffs != 0]
            if len(nonzero) < 10:
                raw_results.append(((m1, m2), 1.0, 0.0))
                continue
            stat, p_val = wilcoxon(abs_e1[:n], abs_e2[:n])
            r = rank_biserial(e1[:n], e2[:n])
            raw_results.append(((m1, m2), p_val, r))

        p_list = [(f"{m1} vs {m2}", p) for (m1, m2), p, _ in raw_results]
        corrected = holm_bonferroni(p_list)

        print(f"  {'Comparison':<45s}  {'p (raw)':>10s}  {'p (adj)':>10s}  {'r_rb':>8s}  {'Sig':>5s}")
        print(f"  {'-'*85}")
        for i, ((m1, m2), raw_p, r) in enumerate(raw_results):
            _, _, adj_p, sig = corrected[i]
            sig_str = "*" if sig else ""
            print(f"  {m1 + ' vs ' + m2:<45s}  {fmt_p(raw_p):>10s}  {fmt_p(adj_p):>10s}  {r:>8.3f}  {sig_str:>5s}")

        n_sig = sum(1 for _, _, _, sig in corrected if sig)
        print(f"\n  {n_sig}/{len(corrected)} comparisons significant after correction")


# ─── 3. ICC(3,k) for intra-model stability ───────────────────────────────────

def icc_3k(ratings):
    """
    ICC(3,k) — two-way mixed, consistency, average measures.
    ratings: (n_subjects, k_raters) array.
    """
    n, k = ratings.shape
    grand_mean = ratings.mean()
    ss_rows = k * np.sum((ratings.mean(axis=1) - grand_mean) ** 2)
    ss_cols = n * np.sum((ratings.mean(axis=0) - grand_mean) ** 2)
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    icc = (ms_rows - ms_error) / ms_rows if ms_rows > 0 else 0.0
    return icc


def report_icc(df_diag, df_treat):
    print_section("3. ICC(3,k) Intra-Model Stability (bootstrap 95% CI)")

    for task_name, df, key in [("Diagnosis", df_diag, "diag"),
                                ("Treatment", df_treat, "treat")]:
        print(f"\n  --- {task_name} ---")
        for m, cols in MODELS.items():
            gt, run_scores = get_per_run_scores(df, cols[key])
            # Each run's error
            errors = np.column_stack([r - gt for r in run_scores])

            def icc_fn(data):
                return icc_3k(data)

            point, lo, hi = bootstrap_ci(errors, icc_fn, n_boot=2000)
            print(f"  {m:<22s}: ICC = {fmt_kappa(point, lo, hi)}")


# ─── 4. Linear weighted kappa for inter-model agreement ──────────────────────

def report_weighted_kappa(df_diag, df_treat):
    print_section("4. Linear Weighted Kappa (Inter-Model Agreement, bootstrap 95% CI)")

    for task_name, df, key in [("Diagnosis", df_diag, "diag"),
                                ("Treatment", df_treat, "treat")]:
        print(f"\n  --- {task_name} ---")

        # Consensus scores discretized to nearest 0.5 for ordinal kappa
        model_consensus = {}
        for m, cols in MODELS.items():
            gt, consensus = get_consensus_scores(df, cols[key])
            # Round to nearest 0.5, convert to string labels for sklearn
            discretized = np.round(consensus * 2) / 2
            model_consensus[m] = np.array([f"{v:.1f}" for v in discretized])

        # Collect all possible labels for consistent kappa computation
        all_labels = sorted(set(
            label for arr in model_consensus.values() for label in arr
        ))

        model_names = list(MODELS.keys())
        for m1, m2 in combinations(model_names, 2):
            s1, s2 = model_consensus[m1], model_consensus[m2]
            n = min(len(s1), len(s2))

            def kappa_fn(data, _labels=all_labels):
                return cohen_kappa_score(data[0], data[1],
                                         labels=_labels, weights="linear")

            point, lo, hi = bootstrap_ci(
                (s1[:n], s2[:n]), kappa_fn, n_boot=2000)
            print(f"  {m1:>22s} vs {m2:<22s}: κ_w = {fmt_kappa(point, lo, hi)}")


# ─── Save CSV ─────────────────────────────────────────────────────────────────

def save_results(df_diag, df_treat, output):
    rows = []
    for task_name, df, key in [("Diagnosis", df_diag, "diag"),
                                ("Treatment", df_treat, "treat")]:
        for m, cols in MODELS.items():
            gt, consensus = get_consensus_scores(df, cols[key])
            errors = consensus - gt
            med = np.median(errors)
            q1, q3 = np.percentile(errors, [25, 75])
            rows.append({
                "task": task_name, "model": m,
                "median_error": round(med, 3),
                "iqr_lo": round(q1, 3), "iqr_hi": round(q3, 3),
                "n": len(errors),
            })
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nSaved: {output}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", default="statistics/nmed_results.csv")
    args = ap.parse_args()

    print("Loading NMED data...")
    df_diag, df_treat = load_nmed()
    print(f"  Diagnosis: {len(df_diag)} cases, Treatment: {len(df_treat)} cases")

    report_error_summary(df_diag, df_treat)
    report_wilcoxon(df_diag, df_treat)
    report_icc(df_diag, df_treat)
    report_weighted_kappa(df_diag, df_treat)
    save_results(df_diag, df_treat, args.output)


if __name__ == "__main__":
    main()
