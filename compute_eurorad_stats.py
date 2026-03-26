#!/usr/bin/env python3
"""
Compute Eurorad benchmark statistics for the updated model lineup.

Reads results/Eurorad.csv and computes:
1. Self-consistency accuracy (majority vote over 3 runs) per model
2. Wilson Score 95% CIs
3. Per-anatomical-category breakdown
4. Pairwise McNemar's test
5. Fleiss' kappa (intra-model stability)
6. Cohen's kappa (inter-model agreement)

Outputs formatted tables + CSV.
"""

import csv
import re
import unicodedata
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import cohen_kappa_score


# ─── Text normalization (matches benchmarks/eurorad/hf_bench.py) ──────────────

def norm_text(s: str) -> str:
    t = unicodedata.normalize("NFKC", s or "")
    t = t.replace("\u2013", "-").replace("\u2014", "-")  # en-dash, em-dash
    t = " ".join(t.strip().split())
    return t.lower()


# ─── Wilson Score CI ──────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return p_hat, lo, hi


def fmt_acc(k, n):
    """Format accuracy with Wilson CI: 'XX.X (XX.X--XX.X)'"""
    p, lo, hi = wilson_ci(k, n)
    return f"{p*100:.1f} ({lo*100:.1f}--{hi*100:.1f})"


# ─── McNemar's test ──────────────────────────────────────────────────────────

def mcnemar_test(correct_a, correct_b):
    """McNemar's test with continuity correction on paired binary outcomes."""
    assert len(correct_a) == len(correct_b)
    b = sum(a and not bb for a, bb in zip(correct_a, correct_b))  # A right, B wrong
    c = sum(not a and bb for a, bb in zip(correct_a, correct_b))  # A wrong, B right
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = 1 - chi2.cdf(stat, df=1)
    return p_val


# ─── Fleiss' Kappa ────────────────────────────────────────────────────────────

def fleiss_kappa(table):
    """
    Fleiss' kappa for inter-rater agreement.
    table: (N_subjects, N_categories) array of counts.
    """
    N, k = table.shape
    n = table.sum(axis=1)[0]  # number of raters per subject
    if n <= 1:
        return float("nan")
    p_j = table.sum(axis=0) / (N * n)
    P_i = (np.sum(table**2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j**2)
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


# ─── Load data ────────────────────────────────────────────────────────────────

def load_eurorad():
    """Load Eurorad.csv handling duplicate DeepSeek column names."""
    with open("results/Eurorad.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_headers = next(reader)
        rows = list(reader)

    # Disambiguate duplicate column names by appending index
    seen = Counter()
    headers = []
    for h in raw_headers:
        count = seen[h]
        seen[h] += 1
        headers.append(h if count == 0 else f"{h}.{count}")

    df = pd.DataFrame(rows, columns=headers)
    # Drop rows with no case_id (empty trailing rows)
    df = df[df["case_id"].str.strip().astype(bool)].copy()

    # Map uncommon sections to "Others"
    OTHERS_SECTIONS = {"Genital (female) imaging", "Interventional radiology"}
    df["Section"] = df["Section"].apply(lambda s: "Others" if s in OTHERS_SECTIONS else s)

    return df


# ─── Model definitions ───────────────────────────────────────────────────────

MODELS = {
    # Proprietary
    "GPT-5.2": ["gpt-5.2-1211-M1", "gpt-5.2-1211-M2", "gpt-5.2-1211-M3"],
    "GPT-5-mini": ["gpt-5-mini-0807-M1", "gpt-5-mini-0807-M2", "gpt-5-mini-0807-M3"],
    "Gemini 3.1 Pro": ["gemini-3.1-pro-M1", "gemini-3.1-pro-M2", "gemini-3.1-pro-M3"],
    # Open
    "DeepSeek-R1": ["deepseek r1 0528 v2", "deepseek r1 0528 v2.1", "deepseek r1 0528 v2.2"],
    # On-device gpt-oss
    "gpt-oss-20b (L)": ["oss-20b (L) v1", "oss-20b (L) v2", "oss-20b (L) v3"],
    "gpt-oss-20b (M)": ["oss-20b (M) v1", "oss-20b (M) v2", "oss-20b (M) v3"],
    "gpt-oss-20b (H)": ["oss-20b (H) v1", "oss-20b (H) v2", "oss-20b (H) v3"],
    "gpt-oss-120b (L)": ["oss-120b (L) v1", "oss-120b (L) v2", "oss-120b (L) v3"],
    "gpt-oss-120b (M)": ["oss-120b (M) v1", "oss-120b (M) v2", "oss-120b (M) v3"],
    "gpt-oss-120b (H)": ["oss-120b (H) v1", "oss-120b (H) v2", "oss-120b (H) v3"],
    # Qwen base
    "Qwen3.5 35B": ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
    "Qwen3.5 27B": ["qwen3.5 27B v1", "qwen3.5 27B v2", "qwen3.5 27B v3"],
    "Qwen3.5 9B": ["qwen3.5 9B v1", "qwen3.5 9B v2", "qwen3.5 9B v3"],
    # Qwen fine-tuned
    "Qwen3.5 35B FT": ["qwen3.5 35B fine-tuned v1", "qwen3.5 35B fine-tuned v2", "qwen3.5 35B fine-tuned v3"],
    "Qwen3.5 9B FT": ["qwen3.5 9B fine-tuned v1", "qwen3.5 9B fine-tuned v2", "qwen3.5 9B fine-tuned v3"],
    # Legacy (for verification)
    "GPT-5": ["gpt-5-0807-M1", "gpt-5-0807-M2", "gpt-5-0807-M3"],
    "o4-mini": ["o4-mini-M1", "o4-mini-M2", "o4-mini-M3"],
    "GPT-5.1": ["gpt-5.1-1113-M1", "gpt-5.1-1113-M2", "gpt-5.1-1113-M3"],
}

# Categories in the order they appear in the paper
CATEGORIES = [
    "Musculoskeletal system",
    "Cardiovascular",
    "Abdominal imaging",
    "Uroradiology & genital male imaging",
    "Neuroradiology",
    "Paediatric radiology",
    "Head & neck imaging",
    "Breast imaging",
    "Chest imaging",
    "Others",
]

# Short names for table display
CAT_SHORT = {
    "Musculoskeletal system": "Musculoskeletal",
    "Cardiovascular": "Cardiovascular",
    "Abdominal imaging": "Abdominal",
    "Uroradiology & genital male imaging": "Uroradiology",
    "Neuroradiology": "Neuroradiology",
    "Paediatric radiology": "Paediatric",
    "Head & neck imaging": "Head & neck",
    "Breast imaging": "Breast",
    "Chest imaging": "Chest",
    "Others": "Others",
}


def compute_majority_vote(df, gt_col="FinalDiagnosis"):
    """
    For each model, compute per-case majority-vote correctness.
    Returns dict: model_name -> list of bool (correct per case).
    Also returns per-run correctness for Fleiss' kappa.
    """
    gt = df[gt_col].apply(norm_text).values
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
            correct = [p == g for p, g in zip(preds, gt)]
            run_correct.append(correct)

        # Majority vote: correct if >=2 of 3 runs are correct
        n_cases = len(gt)
        majority = []
        for i in range(n_cases):
            votes = sum(run_correct[r][i] for r in range(len(cols)))
            majority.append(votes >= 2)

        results[model_name] = majority
        per_run[model_name] = run_correct

    return results, per_run


def compute_category_accuracy(df, majority_results):
    """Compute per-category accuracy with Wilson CIs."""
    sections = df["Section"].values
    all_stats = {}

    for model_name, correct in majority_results.items():
        stats = {}
        # Overall
        n = len(correct)
        k = sum(correct)
        stats["Average"] = (k, n, fmt_acc(k, n))

        # Per category
        for cat in CATEGORIES:
            mask = [s == cat for s in sections]
            n_cat = sum(mask)
            if n_cat == 0:
                # Try partial match
                mask = [cat.lower() in s.lower() for s in sections]
                n_cat = sum(mask)
            k_cat = sum(c for c, m in zip(correct, mask) if m)
            stats[cat] = (k_cat, n_cat, fmt_acc(k_cat, n_cat))

        all_stats[model_name] = stats

    return all_stats


def print_table(stats, model_order, title=""):
    """Print a formatted table."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

    # Header
    header = f"{'Category':<25s}"
    for m in model_order:
        header += f" | {m:>22s}"
    print(header)
    print("-" * len(header))

    # Per category
    for cat in CATEGORIES:
        row = f"{CAT_SHORT.get(cat, cat):<25s}"
        for m in model_order:
            if m in stats:
                _, _, txt = stats[m].get(cat, (0, 0, "N/A"))
                row += f" | {txt:>22s}"
            else:
                row += f" | {'N/A':>22s}"
        print(row)

    # Average
    print("-" * len(header))
    row = f"{'Average':<25s}"
    for m in model_order:
        if m in stats:
            _, _, txt = stats[m]["Average"]
            row += f" | {txt:>22s}"
        else:
            row += f" | {'N/A':>22s}"
    print(row)


def print_mcnemar(majority_results, model_order):
    """Print pairwise McNemar's test p-values."""
    print(f"\n{'='*80}")
    print("  Pairwise McNemar's Test (p-values)")
    print(f"{'='*80}")

    header = f"{'':>22s}"
    for m in model_order:
        header += f" | {m:>15s}"
    print(header)
    print("-" * len(header))

    for m1 in model_order:
        row = f"{m1:>22s}"
        for m2 in model_order:
            if m1 == m2:
                row += f" | {'--':>15s}"
            elif m1 in majority_results and m2 in majority_results:
                p = mcnemar_test(majority_results[m1], majority_results[m2])
                if p < 0.001:
                    row += f" | {'<0.001':>15s}"
                else:
                    row += f" | {p:>15.4f}"
            else:
                row += f" | {'N/A':>15s}"
        print(row)


def compute_fleiss(per_run):
    """Compute Fleiss' kappa per model (binary: correct/incorrect across 3 runs)."""
    print(f"\n{'='*80}")
    print("  Fleiss' Kappa (Intra-Model Stability, 3 runs)")
    print(f"{'='*80}")

    for model_name, runs in per_run.items():
        n_cases = len(runs[0])
        n_raters = len(runs)
        # Build table: (n_cases, 2) where cols = [incorrect, correct]
        table = np.zeros((n_cases, 2), dtype=int)
        for r in range(n_raters):
            for i in range(n_cases):
                if runs[r][i]:
                    table[i, 1] += 1
                else:
                    table[i, 0] += 1
        kappa = fleiss_kappa(table)
        print(f"  {model_name:<25s}: κ = {kappa:.4f}")


def compute_cohen(majority_results, model_order):
    """Compute Cohen's kappa between model pairs (on consensus binary outcomes)."""
    print(f"\n{'='*80}")
    print("  Cohen's Kappa (Inter-Model Agreement on Consensus)")
    print(f"{'='*80}")

    for m1, m2 in combinations(model_order, 2):
        if m1 in majority_results and m2 in majority_results:
            labels_a = [int(c) for c in majority_results[m1]]
            labels_b = [int(c) for c in majority_results[m2]]
            kappa = cohen_kappa_score(labels_a, labels_b)
            print(f"  {m1:>22s} vs {m2:<22s}: κ = {kappa:.4f}")


def save_csv(stats, model_order, filename="results/eurorad_stats.csv"):
    """Save all accuracy stats to CSV."""
    rows = []
    for cat in CATEGORIES + ["Average"]:
        row = {"Category": CAT_SHORT.get(cat, cat)}
        for m in model_order:
            if m in stats:
                k, n, txt = stats[m].get(cat, (0, 0, "N/A"))
                p, lo, hi = wilson_ci(k, n) if n > 0 else (0, 0, 0)
                row[f"{m}_acc"] = f"{p*100:.1f}"
                row[f"{m}_ci"] = txt
                row[f"{m}_k"] = k
                row[f"{m}_n"] = n
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(filename, index=False)
    print(f"\nSaved to {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading Eurorad data...")
    df = load_eurorad()
    print(f"  {len(df)} cases loaded, {len(df.columns)} columns")

    # Filter to rows with data (first 207-208 rows have predictions)
    gt_filled = df["FinalDiagnosis"].apply(lambda x: bool(str(x).strip()))
    df = df[gt_filled].copy()
    print(f"  {len(df)} cases with ground truth")

    # Check categories
    cats = df["Section"].value_counts()
    print(f"\n  Anatomical categories:")
    for cat in CATEGORIES:
        count = cats.get(cat, 0)
        print(f"    {cat}: {count}")

    print("\nComputing majority vote...")
    majority, per_run = compute_majority_vote(df)

    print("\nComputing per-category accuracy...")
    stats = compute_category_accuracy(df, majority)

    # ── New model lineup ──
    new_proprietary = ["GPT-5.2", "GPT-5-mini", "Gemini 3.1 Pro"]
    new_open = ["DeepSeek-R1"]
    new_oss20b = ["gpt-oss-20b (L)", "gpt-oss-20b (M)", "gpt-oss-20b (H)"]
    new_oss120b = ["gpt-oss-120b (L)", "gpt-oss-120b (M)", "gpt-oss-120b (H)"]
    new_qwen = ["Qwen3.5 35B", "Qwen3.5 27B", "Qwen3.5 9B"]
    new_qwen_ft = ["Qwen3.5 35B FT", "Qwen3.5 9B FT"]
    legacy = ["GPT-5", "o4-mini", "GPT-5.1"]

    new_order = new_proprietary + new_open + new_oss20b + new_oss120b + new_qwen
    all_order = new_order + new_qwen_ft + legacy

    # Print tables
    print_table(stats, new_order, "Table 1: LLM-as-a-Generalist (Updated Model Lineup)")
    print_table(stats, new_qwen_ft + ["gpt-oss-20b (M)"], "Fine-tuned Models")
    print_table(stats, legacy, "Legacy Models (for verification)")

    # Supplementary: full CIs (same data, just printed differently)
    print(f"\n{'='*80}")
    print("  Supplementary Table 1: Full CIs")
    print(f"{'='*80}")
    for m in all_order:
        if m not in stats:
            continue
        k, n, txt = stats[m]["Average"]
        print(f"  {m:<25s}: {txt}  (k={k}, n={n})")

    # McNemar's
    print_mcnemar(majority, new_order)

    # Fleiss' kappa
    compute_fleiss(per_run)

    # Cohen's kappa
    compute_cohen(majority, new_proprietary + new_open + ["gpt-oss-120b (L)", "gpt-oss-20b (H)"] + ["Qwen3.5 35B"])

    # Save CSV
    save_csv(stats, all_order)

    # ── LaTeX snippet for Table 1 ──
    print(f"\n{'='*80}")
    print("  LaTeX Table 1 Rows (copy-paste ready)")
    print(f"{'='*80}")
    table_models = new_proprietary + new_open + new_oss20b + new_oss120b + new_qwen
    for cat in CATEGORIES:
        parts = []
        for m in table_models:
            if m in stats:
                k, n, _ = stats[m].get(cat, (0, 0, ""))
                p, _, _ = wilson_ci(k, n) if n > 0 else (0, 0, 0)
                parts.append(f"\\acc{{{p*100:.1f}}}")
            else:
                parts.append("--")
        short = CAT_SHORT.get(cat, cat)
        line = f"{short:<20s} & " + " & ".join(parts) + " \\\\"
        print(line)

    # Average row
    parts = []
    for m in table_models:
        if m in stats:
            k, n, _ = stats[m]["Average"]
            p, _, _ = wilson_ci(k, n)
            parts.append(f"\\best{{\\acc{{{p*100:.1f}}}}}" if False else f"\\acc{{{p*100:.1f}}}")
    print(f"{'Average':<20s} & " + " & ".join(parts) + " \\\\")


if __name__ == "__main__":
    main()
