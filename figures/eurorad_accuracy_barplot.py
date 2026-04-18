#!/usr/bin/env python3
"""
Eurorad accuracy bar plot with Wilson score 95% CIs.

Computes self-consistency accuracy (majority vote over 3 runs) per model,
then plots grouped bars with error bars. Models are grouped by category
(Proprietary, Open, On-device OSS, On-device Qwen).

Usage:
    python figures/eurorad_accuracy_barplot.py
    python figures/eurorad_accuracy_barplot.py --output figures/eurorad_accuracy.pdf
"""

import csv
import re
import argparse
import unicodedata
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from palette import MODEL_COLORS


# ─── Text normalization (matches benchmarks/eurorad/hf_bench.py) ────────────

def norm_text(s: str) -> str:
    t = s or ""
    try:
        t = t.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[^\w\s-]", "", t)
    t = " ".join(t.strip().split())
    return t.lower()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0.0, centre - margin), min(1.0, centre + margin)


# ─── Model definitions ──────────────────────────────────────────────────────

MODELS = {
    "DeepSeek-R1": ["deepseek r1 0528 v2", "deepseek r1 0528 v2.1", "deepseek r1 0528 v2.2"],
    "GPT-5-mini": ["gpt-5-mini-0807-M1", "gpt-5-mini-0807-M2", "gpt-5-mini-0807-M3"],
    "GPT-5.1": ["gpt-5.1-1113-M1", "gpt-5.1-1113-M2", "gpt-5.1-1113-M3"],
    "Gemini 3.1 Pro": ["gemini-3.1-pro-M1", "gemini-3.1-pro-M2", "gemini-3.1-pro-M3"],
    # On-device OSS (high reasoning only)
    "OSS-20B (H)": ["oss-20b (H) v1", "oss-20b (H) v2", "oss-20b (H) v3"],
    "OSS-120B (H)": ["oss-120b (H) v1", "oss-120b (H) v2", "oss-120b (H) v3"],
    # Qwen (base only)
    "Qwen3.5 9B": ["qwen3.5 9B v1", "qwen3.5 9B v2", "qwen3.5 9B v3"],
    "Qwen3.5 27B": ["qwen3.5 27B v1", "qwen3.5 27B v2", "qwen3.5 27B v3"],
    "Qwen3.5 35B": ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
    "Gemma 4 31B": ["gemma-4-31b-M1", "gemma-4-31b-M2", "gemma-4-31b-M3"],
}

MODEL_ORDER = list(MODELS.keys())

SHORT_LABELS = {
    "DeepSeek-R1": "DeepSeek-R1",
    "GPT-5-mini": "GPT-5-mini",
    "GPT-5.1": "GPT-5.1",
    "Gemini 3.1 Pro": "Gemini 3.1",
    "OSS-20B (H)": "20B",
    "OSS-120B (H)": "120B",
    "Qwen3.5 9B": "9B",
    "Qwen3.5 27B": "27B",
    "Qwen3.5 35B": "35B",
    "Gemma 4 31B": "31B",
}

GROUPS = {
    "Proprietary": ["DeepSeek-R1", "GPT-5-mini", "GPT-5.1", "Gemini 3.1 Pro"],
    "gpt-oss": ["OSS-20B (H)", "OSS-120B (H)"],
    "Qwen 3.5": ["Qwen3.5 9B", "Qwen3.5 27B", "Qwen3.5 35B"],
    "Gemma 4": ["Gemma 4 31B"],
}

# ─── Load & compute ─────────────────────────────────────────────────────────

def load_eurorad(path="csvs/final_csvs/Eurorad.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_headers = next(reader)
        rows = list(reader)

    seen = Counter()
    headers = []
    for h in raw_headers:
        count = seen[h]
        seen[h] += 1
        headers.append(h if count == 0 else f"{h}.{count}")

    data = {h: [] for h in headers}
    for row in rows:
        for h, val in zip(headers, row):
            data[h].append(val)

    # Filter rows with case_id
    valid = [i for i, cid in enumerate(data["case_id"]) if cid.strip()]
    return headers, {h: [data[h][i] for i in valid] for h in headers}


def compute_accuracy(headers, data):
    gt = [norm_text(s) for s in data["FinalDiagnosis"]]
    n_cases = len(gt)
    results = {}

    for model_name in MODEL_ORDER:
        cols = MODELS[model_name]
        missing = [c for c in cols if c not in headers]
        if missing:
            continue

        run_correct = []
        for col in cols:
            preds = [norm_text(s) for s in data[col]]
            run_correct.append([p == g for p, g in zip(preds, gt)])

        # Majority vote: correct if >= 2 of 3 runs match
        k = sum(
            sum(run_correct[r][i] for r in range(len(cols))) >= 2
            for i in range(n_cases)
        )
        acc, lo, hi = wilson_ci(k, n_cases)
        results[model_name] = (acc, lo, hi, k, n_cases)

    return results


# ─── Plot ────────────────────────────────────────────────────────────────────

def plot_accuracy(results, output=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Build bar positions with gaps between groups
    positions = []
    colors = []
    labels = []
    gap = 1.2
    x = 0
    group_spans = {}

    for group_name, model_names in GROUPS.items():
        start = x
        for m in model_names:
            if m in results:
                positions.append(x)
                colors.append(MODEL_COLORS[m])
                labels.append(m)
                x += 1
        group_spans[group_name] = (start, x - 1)
        x += gap

    accs = [results[m][0] * 100 for m in labels]
    los = [results[m][1] * 100 for m in labels]
    his = [results[m][2] * 100 for m in labels]
    err_lo = [a - l for a, l in zip(accs, los)]
    err_hi = [h - a for a, h in zip(accs, his)]

    ax.bar(
        positions, accs,
        yerr=[err_lo, err_hi],
        capsize=4,
        color=colors,
        edgecolor="none",
        width=0.72,
        error_kw={"linewidth": 1.2, "color": "black", "capthick": 1.2},
    )

    # Value labels above bars with white background
    for pos, acc in zip(positions, accs):
        ax.text(pos, acc + 1.5, f"{acc:.1f}", ha="center", va="center",
                fontsize=8, color="#333333", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))

    # Group bracket labels
    trans = ax.get_xaxis_transform()
    for group_name, (start, end) in group_spans.items():
        mid = (start + end) / 2
        if start == end:
            ax.text(mid, -0.18, group_name, ha="center", va="top",
                    fontsize=9, color="#666666", transform=trans)
        else:
            ax.plot([start - 0.35, end + 0.35], [-0.18, -0.18],
                    color="#999999", lw=0.8, transform=trans, clip_on=False)
            ax.text(mid, -0.22, group_name, ha="center", va="top",
                    fontsize=9, color="#666666", transform=trans)

    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT_LABELS[m] for m in labels], fontsize=10)
    ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=12)
    ax.set_ylim(60, 95)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, color="#cccccc", linewidth=0.5)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(bottom=0.22)

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="csvs/final_csvs/Eurorad.csv")
    ap.add_argument("--output", "-o", default=None,
                    help="Output file (e.g., figures/eurorad_accuracy.png). Shows interactive plot if omitted.")
    args = ap.parse_args()

    headers, data = load_eurorad(args.input)
    results = compute_accuracy(headers, data)

    print(f"{'Model':<20s}  {'Acc':>6s}  {'95% CI':>15s}  {'k/n':>7s}")
    print("-" * 55)
    for m in MODEL_ORDER:
        if m in results:
            acc, lo, hi, k, n = results[m]
            print(f"{m:<20s}  {acc*100:5.1f}%  ({lo*100:4.1f} - {hi*100:4.1f})  {k:3d}/{n}")

    plot_accuracy(results, args.output)


if __name__ == "__main__":
    main()
