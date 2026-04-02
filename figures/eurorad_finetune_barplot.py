#!/usr/bin/env python3
"""
Eurorad fine-tuning improvement bar plot.

Shows proprietary model references alongside base → fine-tuned pairs
for gpt-oss-120B and Qwen 3.5 models, with bracket annotations for
the accuracy delta.

Usage:
    python figures/eurorad_finetune_barplot.py
    python figures/eurorad_finetune_barplot.py --output figures/eurorad_finetune.png
"""

import csv
import re
import argparse
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from palette import MODEL_COLORS


def norm_text(s: str) -> str:
    t = str(s) if s is not None and str(s).strip() else ""
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


def majority_vote_accuracy(gt_series, col_list, df):
    gt = gt_series.apply(norm_text)
    mask = gt.astype(bool)
    n = mask.sum()
    run_correct = []
    for col in col_list:
        preds = df[col].apply(norm_text)
        run_correct.append((preds[mask] == gt[mask]).values)
    maj_k = sum(sum(run_correct[r][i] for r in range(len(col_list))) >= 2
                for i in range(n))
    acc, lo, hi = wilson_ci(maj_k, n)
    return acc * 100, lo * 100, hi * 100


def load_eurorad_results():
    """Load accuracy from main Eurorad CSV."""
    df = pd.read_csv("csvs/Eurorad.csv")
    df = df[df["FinalDiagnosis"].notna() & (df["FinalDiagnosis"].astype(str).str.strip() != "")]
    gt = df["FinalDiagnosis"]

    models = {
        "GPT-5.1": ["gpt-5.1-1113-M1", "gpt-5.1-1113-M2", "gpt-5.1-1113-M3"],
        "GPT-5-mini": ["gpt-5-mini-0807-M1", "gpt-5-mini-0807-M2", "gpt-5-mini-0807-M3"],
        "Gemini 3.1": ["gemini-3.1-pro-M1", "gemini-3.1-pro-M2", "gemini-3.1-pro-M3"],
        "DeepSeek-R1": ["deepseek r1 0528 v2", "deepseek r1 0528 v2.1", "deepseek r1 0528 v2.2"],
        "Qwen 35B": ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
        "35B (FT)": ["qwen3.5 35B fine-tuned v1", "qwen3.5 35B fine-tuned v2", "qwen3.5 35B fine-tuned v3"],
    }

    results = {}
    for name, cols in models.items():
        results[name] = majority_vote_accuracy(gt, cols, df)
    return results


def load_finetune_eval_results():
    """Load OSS-120B base and best FT checkpoint from finetune-eval."""
    ft_dir = "csvs/finetune-eval"

    checkpoints = {
        "gpt-oss-120B": [
            f"{ft_dir}/oss120b_base_eurorad.csv",
            f"{ft_dir}/oss120b_base_eurorad_v2.csv",
            f"{ft_dir}/oss120b_base_eurorad_v3.csv",
        ],
        "120B (FT)": [
            f"{ft_dir}/oss120b_r32_a64_attn_552_eurorad.csv",
            f"{ft_dir}/oss120b_r32_a64_attn_552_eurorad_v2.csv",
            f"{ft_dir}/oss120b_r32_a64_attn_552_eurorad_v3.csv",
        ],
    }

    results = {}
    for name, files in checkpoints.items():
        run_correct = []
        n_cases = None
        for f in files:
            df = pd.read_csv(f)
            gt = df["FinalDiagnosis"].apply(norm_text)
            pred = df["model_answer"].apply(norm_text)
            mask = gt.astype(bool)
            correct = (gt[mask] == pred[mask]).values
            run_correct.append(correct)
            if n_cases is None:
                n_cases = len(correct)

        maj_k = sum(sum(run_correct[r][i] for r in range(3)) >= 2
                    for i in range(n_cases))
        acc, lo, hi = wilson_ci(maj_k, n_cases)
        results[name] = (acc * 100, lo * 100, hi * 100)

    return results


def draw_bracket(ax, x1, x2, y, delta_text, color="#555555"):
    """Draw a bracket with delta annotation between two bars."""
    bracket_h = 0.8
    ax.plot([x1, x1, x2, x2], [y, y + bracket_h, y + bracket_h, y],
            color=color, lw=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + bracket_h + 0.3, delta_text,
            ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")


def plot(output=None):
    euro_results = load_eurorad_results()
    ft_results = load_finetune_eval_results()

    # Bar order and properties
    bar_names = [
        "DeepSeek-R1", "GPT-5-mini", "GPT-5.1", "Gemini 3.1",
        "Qwen 35B", "35B (FT)",
    ]
    bars = [(name, MODEL_COLORS[name]) for name in bar_names]

    all_results = {**euro_results, **ft_results}

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Positions with gaps between groups
    positions = []
    x = 0
    gaps_after = {3}  # after Gemini
    for i in range(len(bars)):
        positions.append(x)
        x += 1
        if i in gaps_after:
            x += 0.8

    accs, los, his, colors, labels = [], [], [], [], []
    for label, color in bars:
        acc, lo, hi = all_results[label]
        accs.append(acc)
        los.append(lo)
        his.append(hi)
        colors.append(color)
        labels.append(label)

    err_lo = [a - l for a, l in zip(accs, los)]
    err_hi = [h - a for a, h in zip(accs, his)]

    ax.bar(positions, accs,
           yerr=[err_lo, err_hi],
           capsize=4, color=colors, edgecolor="none", width=0.72,
           error_kw={"linewidth": 1.2, "color": "black", "capthick": 1.2})

    # Value labels above bars with white background
    for pos, acc in zip(positions, accs):
        ax.text(pos, acc + 1.5, f"{acc:.1f}", ha="center", va="center",
                fontsize=8, color="#333333", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))

    # Bracket annotations for FT pairs
    ft_pairs = [
        (4, 5),  # Qwen 35B → FT
    ]
    for base_idx, ft_idx in ft_pairs:
        delta = accs[ft_idx] - accs[base_idx]
        bracket_y = max(his[base_idx], his[ft_idx]) + 0.5
        draw_bracket(ax, positions[base_idx], positions[ft_idx],
                     bracket_y, f"+{delta:.1f}%")

    # Group labels
    trans = ax.get_xaxis_transform()
    groups = [
        ("Proprietary", 0, 3),
        ("Qwen 3.5", 4, 5),
    ]
    for group_name, i_start, i_end in groups:
        mid = (positions[i_start] + positions[i_end]) / 2
        ax.plot([positions[i_start] - 0.35, positions[i_end] + 0.35],
                [-0.20, -0.20], color="#999999", lw=0.8,
                transform=trans, clip_on=False)
        ax.text(mid, -0.24, group_name, ha="center", va="top",
                fontsize=9, color="#666666", transform=trans)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=12)
    ax.set_ylim(60, 100)
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
    ap.add_argument("--output", "-o", default=None)
    args = ap.parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()
