#!/usr/bin/env python3
"""
Phase 1.3 — Error taxonomy on the hard tail.

Classifies every incorrect k=3 majority prediction into 5 categories using
Claude Opus 4.6 as a neutral judge (not in the evaluation set).

Categories:
  (a) Correct, surface-form mismatch (synonym, eponym, nomenclature variant)
  (b) Same disease family, wrong specific entity
  (c) Plausible differential, incorrect
  (d) Off-topic or hallucinated (not in provided differential list)
  (e) Empty, refusal, or parse failure

Usage:
    python analysis/error_taxonomy.py
    python analysis/error_taxonomy.py --dry-run       # show errors without calling API
    python analysis/error_taxonomy.py --resume         # skip already-classified rows
    python analysis/error_taxonomy.py --sample 50      # classify only N errors (for testing)
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import boto3
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

MODEL_FAMILIES = {
    "GPT-5.1": "proprietary",
    "GPT-5-mini": "proprietary",
    "Gemini 3.1 Pro": "proprietary",
    "DeepSeek-R1": "open_large",
    "gpt-oss-20b (H)": "on_device_base",
    "gpt-oss-120b (H)": "on_device_base",
    "Qwen3.5 9B": "on_device_base",
    "Qwen3.5 27B": "on_device_base",
    "Qwen3.5 35B": "on_device_base",
    "Qwen3.5 35B FT": "on_device_ft",
    "Gemma 4 31B": "on_device_base",
}

OTHERS_SECTIONS = {"Genital (female) imaging", "Interventional radiology"}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

RUBRIC_PROMPT = """You are an expert medical diagnostician classifying errors in a clinical diagnosis task.

A model was given a patient case and a list of differential diagnoses, and asked to select the correct diagnosis verbatim from the list. The model's prediction was scored INCORRECT under exact text matching.

Your task: classify this error into exactly one of five categories.

## Categories

(a) SURFACE_MISMATCH — The prediction is clinically correct but differs in surface form (synonym, eponym, abbreviation, nomenclature variant, or minor wording difference). The model identified the right condition.

(b) SAME_FAMILY — The prediction names a condition in the same disease family or anatomical system as the ground truth, but is the wrong specific entity (e.g., "angiosarcoma" when the answer is "cardiac lymphoma" — both cardiac tumors).

(c) PLAUSIBLE_DIFFERENTIAL — The prediction is a plausible differential diagnosis that was on the provided list, but is incorrect. The model's reasoning was clinically defensible but wrong.

(d) OFF_TOPIC — The prediction names a condition that is not on the differential list, is anatomically/pathologically unrelated, or is a hallucinated entity.

(e) EMPTY_OR_REFUSAL — The prediction is empty, a refusal to answer, a parse failure, or contains no identifiable diagnosis.

## Case information

Ground truth diagnosis: {ground_truth}
Model prediction: {prediction}
Differential diagnosis list: {differentials}

## Instructions

Respond with ONLY the category letter (a, b, c, d, or e) on the first line, followed by a one-sentence justification on the second line. Example:
b
The prediction "angiosarcoma" is in the same organ system (cardiac tumors) as the ground truth "primary cardiac lymphoma" but is the wrong specific entity."""


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


def get_majority_prediction(df, cols, i):
    """Get the majority-vote prediction text for case i."""
    preds = [norm_text(df.iloc[i][col]) for col in cols]
    counts = Counter(preds)
    majority_pred, _ = counts.most_common(1)[0]
    return majority_pred


def get_differentials(df, i):
    """Extract differential diagnosis list from the CSV."""
    for col in ["DifferentialDiagnosisList", "differentials", "Differentials"]:
        if col in df.columns:
            val = str(df.iloc[i][col]).strip()
            if val and val != "nan":
                return val
    return "Not available"


def collect_errors(df):
    """Find all (model, case_idx) pairs where majority vote is incorrect."""
    gt = df["FinalDiagnosis"].apply(norm_text).values
    errors = []

    for model_name, cols in MODELS.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            continue

        for i in range(len(df)):
            majority_pred = get_majority_prediction(df, cols, i)
            if majority_pred != gt[i]:
                raw_pred = get_majority_prediction(df, cols, i)
                # Get raw (unnormalized) prediction for display
                raw_preds = [str(df.iloc[i][col]).strip() for col in cols]
                raw_counts = Counter(raw_preds)
                display_pred, _ = raw_counts.most_common(1)[0]

                errors.append({
                    "case_id": df.iloc[i]["case_id"],
                    "case_idx": i,
                    "model": model_name,
                    "family": MODEL_FAMILIES[model_name],
                    "section": df.iloc[i]["Section"],
                    "ground_truth": df.iloc[i]["FinalDiagnosis"],
                    "prediction": display_pred,
                    "prediction_normalized": majority_pred,
                    "differentials": get_differentials(df, i),
                })

    return errors


def classify_error(client, error, model_id="us.anthropic.claude-opus-4-6-v1"):
    """Call Claude to classify a single error."""
    prompt = RUBRIC_PROMPT.format(
        ground_truth=error["ground_truth"],
        prediction=error["prediction"],
        differentials=error["differentials"],
    )

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )

    result = json.loads(response["body"].read())
    text = result["content"][0]["text"].strip()

    lines = text.split("\n", 1)
    category = lines[0].strip().lower().rstrip(".")
    justification = lines[1].strip() if len(lines) > 1 else ""

    valid_categories = {"a", "b", "c", "d", "e"}
    if category not in valid_categories:
        for c in valid_categories:
            if c in category:
                category = c
                break
        else:
            category = "unknown"

    return category, justification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Collect errors without calling API")
    ap.add_argument("--resume", action="store_true", help="Skip already-classified rows")
    ap.add_argument("--sample", type=int, default=0, help="Classify only N errors (for testing)")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-6-v1", help="Bedrock model ID")
    args = ap.parse_args()

    print("Loading Eurorad data...")
    df = load_eurorad()
    df = df[df["FinalDiagnosis"].apply(lambda x: bool(str(x).strip()))].copy()
    print(f"  {len(df)} cases with ground truth")

    print("Collecting errors...")
    errors = collect_errors(df)
    print(f"  {len(errors)} total errors across all models")

    if args.dry_run:
        error_df = pd.DataFrame(errors)
        print(f"\n  Errors per model:")
        for m, count in error_df["model"].value_counts().items():
            print(f"    {m}: {count}")
        print(f"\n  Errors per family:")
        for f, count in error_df["family"].value_counts().items():
            print(f"    {f}: {count}")
        out_path = OUTPUT_DIR / "errors_unclassified.csv"
        error_df.to_csv(out_path, index=False)
        print(f"\n  Saved unclassified errors: {out_path}")
        return

    # Load existing results if resuming
    out_path = OUTPUT_DIR / "error_taxonomy.csv"
    classified = set()
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        classified = set(zip(existing["case_id"].astype(str), existing["model"]))
        print(f"  Resuming: {len(classified)} already classified")

    # Filter to unclassified
    to_classify = [e for e in errors if (str(e["case_id"]), e["model"]) not in classified]
    if args.sample > 0:
        to_classify = to_classify[:args.sample]
    print(f"  Classifying {len(to_classify)} errors...")

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    results = []

    for idx, error in enumerate(to_classify):
        try:
            category, justification = classify_error(client, error, model_id=args.model)
        except Exception as e:
            print(f"  ERROR on {error['case_id']}/{error['model']}: {e}")
            category, justification = "error", str(e)
            time.sleep(5)

        error["category"] = category
        error["justification"] = justification
        results.append(error)

        if (idx + 1) % 25 == 0:
            print(f"  Classified {idx + 1}/{len(to_classify)}")
            # Incremental save
            _save_results(results, classified, out_path, errors)

    _save_results(results, classified, out_path, errors)
    print(f"\nSaved: {out_path}")

    # Summary
    result_df = pd.read_csv(out_path)
    print(f"\n--- Error Taxonomy Summary ---")
    print(f"  Total classified: {len(result_df)}")
    print(f"\n  Category distribution:")
    for cat, count in result_df["category"].value_counts().items():
        pct = count / len(result_df) * 100
        print(f"    ({cat}): {count} ({pct:.1f}%)")

    print(f"\n  By model family:")
    for fam in ["proprietary", "open_large", "on_device_base", "on_device_ft"]:
        fam_df = result_df[result_df["family"] == fam]
        if fam_df.empty:
            continue
        print(f"\n    {fam} ({len(fam_df)} errors):")
        for cat, count in fam_df["category"].value_counts().items():
            pct = count / len(fam_df) * 100
            print(f"      ({cat}): {count} ({pct:.1f}%)")

    # Save per-family aggregated percentages
    agg_rows = []
    for fam in ["proprietary", "open_large", "on_device_base", "on_device_ft"]:
        fam_df = result_df[result_df["family"] == fam]
        if fam_df.empty:
            continue
        total = len(fam_df)
        row = {"family": fam, "total_errors": total}
        for cat in ["a", "b", "c", "d", "e"]:
            count = (fam_df["category"] == cat).sum()
            row[f"cat_{cat}_count"] = count
            row[f"cat_{cat}_pct"] = round(count / total * 100, 1)
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_path = OUTPUT_DIR / "error_taxonomy_by_family.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"\nSaved: {agg_path}")


def _save_results(new_results, classified_set, out_path, all_errors):
    """Save results, merging with any existing file."""
    new_df = pd.DataFrame(new_results)
    if out_path.exists() and classified_set:
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    # Drop case_idx before saving
    save_cols = [c for c in combined.columns if c != "case_idx"]
    combined[save_cols].to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
