#!/usr/bin/env python3
"""
Phase 1.1 — Case-level difficulty analysis.

For each of 207 Eurorad cases, computes what fraction of models got it right
(k=3 majority vote), cross-tabbed by model family. Identifies:
  - hard-for-all cases (bottom decile across all families)
  - capability-gap cases (correct by proprietary/FT, wrong by base on-device)

Usage:
    python analysis/case_difficulty.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "statistics"))
from utils import norm_text


MODELS = {
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
    "Gemma 4 31B":      ["gemma-4-31b-M1", "gemma-4-31b-M2", "gemma-4-31b-M3"],
}

FAMILIES = {
    "proprietary": ["GPT-5.1", "GPT-5-mini", "Gemini 3.1 Pro"],
    "open_large":  ["DeepSeek-R1"],
    "on_device_base": ["gpt-oss-20b (H)", "gpt-oss-120b (H)", "Qwen3.5 9B", "Qwen3.5 27B", "Qwen3.5 35B", "Gemma 4 31B"],
    "on_device_ft": ["Qwen3.5 35B FT"],
}

OTHERS_SECTIONS = {"Genital (female) imaging", "Interventional radiology"}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_eurorad():
    csv_path = Path(__file__).resolve().parent.parent / "csvs" / "final_csvs" / "Eurorad.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
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


def compute_per_case_majority(df):
    """Returns dict: model_name -> list of bool (correct per case)."""
    gt = df["FinalDiagnosis"].apply(norm_text).values
    results = {}
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
    return results


def main():
    print("Loading Eurorad data...")
    df = load_eurorad()
    df = df[df["FinalDiagnosis"].apply(lambda x: bool(str(x).strip()))].copy()
    print(f"  {len(df)} cases with ground truth")

    majority = compute_per_case_majority(df)
    model_names = [m for m in MODELS if m in majority]
    n_models = len(model_names)
    n_cases = len(df)

    # Build per-case difficulty table
    rows = []
    for i in range(n_cases):
        case_id = df.iloc[i]["case_id"]
        section = df.iloc[i]["Section"]
        gt = df.iloc[i]["FinalDiagnosis"]

        n_correct = sum(majority[m][i] for m in model_names)
        frac_correct = n_correct / n_models

        family_correct = {}
        for fam, members in FAMILIES.items():
            active = [m for m in members if m in majority]
            if active:
                family_correct[fam] = sum(majority[m][i] for m in active) / len(active)
            else:
                family_correct[fam] = np.nan

        row = {
            "case_id": case_id,
            "section": section,
            "ground_truth": gt,
            "n_models_correct": n_correct,
            "n_models_total": n_models,
            "frac_correct": round(frac_correct, 3),
        }
        for fam in FAMILIES:
            row[f"frac_{fam}"] = round(family_correct[fam], 3) if not np.isnan(family_correct[fam]) else np.nan
        for m in model_names:
            row[m] = int(majority[m][i])
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("frac_correct")

    # Save full table
    out_path = OUTPUT_DIR / "case_difficulty.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(result_df)} cases)")

    # Hard-for-all: bottom decile by frac_correct
    threshold = np.percentile(result_df["frac_correct"], 10)
    hard_all = result_df[result_df["frac_correct"] <= threshold]
    hard_path = OUTPUT_DIR / "hard_for_all.csv"
    hard_all.to_csv(hard_path, index=False)
    print(f"Saved: {hard_path} ({len(hard_all)} cases, threshold <= {threshold:.2f})")

    # Capability gap: correct by proprietary OR FT, wrong by all base on-device
    gap_mask = (
        ((result_df["frac_proprietary"] > 0.5) | (result_df["frac_on_device_ft"] > 0.5))
        & (result_df["frac_on_device_base"] < 0.5)
    )
    gap_cases = result_df[gap_mask]
    gap_path = OUTPUT_DIR / "capability_gap.csv"
    gap_cases.to_csv(gap_path, index=False)
    print(f"Saved: {gap_path} ({len(gap_cases)} cases)")

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"  Total cases: {n_cases}")
    print(f"  Models: {n_models}")
    print(f"  Mean difficulty (frac models correct): {result_df['frac_correct'].mean():.3f}")
    print(f"  Hard-for-all cases (bottom decile): {len(hard_all)}")
    print(f"  Capability-gap cases: {len(gap_cases)}")

    # Distribution
    bins = [0, 0.25, 0.5, 0.75, 1.01]
    labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    result_df["difficulty_bin"] = pd.cut(result_df["frac_correct"], bins=bins, labels=labels, right=False)
    print(f"\n  Difficulty distribution:")
    for label in labels:
        count = (result_df["difficulty_bin"] == label).sum()
        print(f"    {label}: {count} cases ({count/n_cases*100:.1f}%)")


if __name__ == "__main__":
    main()
