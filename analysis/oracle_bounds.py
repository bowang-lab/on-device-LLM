#!/usr/bin/env python3
"""
Phase 1.2 — Best-of-N oracle bounds.

For each model, computes three accuracy levels on the 207 Eurorad cases:
  - Single-run accuracy: mean accuracy across the 3 individual runs
  - Majority-vote accuracy: correct if >= 2/3 runs match ground truth
  - Best-of-3 oracle: correct if ANY of the 3 runs matches ground truth

The gap between majority and oracle quantifies recoverable performance —
cases where the model "knows" the answer but sampling noise loses it.

Usage:
    python analysis/oracle_bounds.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "statistics"))
from utils import norm_text, wilson_ci


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


def main():
    print("Loading Eurorad data...")
    df = load_eurorad()
    df = df[df["FinalDiagnosis"].apply(lambda x: bool(str(x).strip()))].copy()
    n_cases = len(df)
    print(f"  {n_cases} cases with ground truth")

    gt = df["FinalDiagnosis"].apply(norm_text).values

    rows = []
    violations = []

    for model_name, cols in MODELS.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  WARNING: {model_name} missing columns: {missing}")
            continue

        run_correct = []
        for col in cols:
            preds = df[col].apply(norm_text).values
            run_correct.append([p == g for p, g in zip(preds, gt)])

        # Single-run: mean accuracy across runs
        single_run_accs = [sum(rc) / n_cases for rc in run_correct]
        single_run_mean = sum(single_run_accs) / len(single_run_accs)

        # Majority vote: correct if >= 2/3 runs match
        majority_correct = [
            sum(run_correct[r][i] for r in range(len(cols))) >= 2
            for i in range(n_cases)
        ]
        majority_k = sum(majority_correct)

        # Best-of-3 oracle: correct if ANY run matches
        oracle_correct = [
            any(run_correct[r][i] for r in range(len(cols)))
            for i in range(n_cases)
        ]
        oracle_k = sum(oracle_correct)

        single_pct = single_run_mean * 100
        majority_pct = majority_k / n_cases * 100
        oracle_pct = oracle_k / n_cases * 100

        # Oracle gap: cases recoverable by better sampling
        oracle_gap = oracle_k - majority_k

        _, maj_lo, maj_hi = wilson_ci(majority_k, n_cases)
        _, orc_lo, orc_hi = wilson_ci(oracle_k, n_cases)

        rows.append({
            "model": model_name,
            "single_run_pct": round(single_pct, 1),
            "majority_pct": round(majority_pct, 1),
            "majority_ci_lo": round(maj_lo * 100, 1),
            "majority_ci_hi": round(maj_hi * 100, 1),
            "majority_k": majority_k,
            "oracle_pct": round(oracle_pct, 1),
            "oracle_ci_lo": round(orc_lo * 100, 1),
            "oracle_ci_hi": round(orc_hi * 100, 1),
            "oracle_k": oracle_k,
            "oracle_gap": oracle_gap,
            "n": n_cases,
        })

        # Acceptance check: oracle >= majority (single_run can exceed majority
        # when a model gets 1/3 correct — counts 0 for majority but 0.33 for mean)
        if oracle_pct < majority_pct - 0.05:
            violations.append(f"  {model_name}: oracle ({oracle_pct:.1f}) < majority ({majority_pct:.1f})")

    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "oracle_bounds.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Print table
    print(f"\n{'Model':<22s}  {'Single':>7s}  {'Majority':>9s}  {'Oracle':>7s}  {'Gap':>5s}")
    print(f"{'-'*55}")
    for _, r in result_df.iterrows():
        print(f"{r['model']:<22s}  {r['single_run_pct']:>6.1f}%  {r['majority_pct']:>8.1f}%  {r['oracle_pct']:>6.1f}%  {r['oracle_gap']:>5.0f}")

    if violations:
        print(f"\n  ACCEPTANCE VIOLATIONS:")
        for v in violations:
            print(v)
    else:
        print(f"\n  Acceptance check passed: oracle >= majority >= single_run for all models")


if __name__ == "__main__":
    main()
