#!/usr/bin/env python3
"""
Eurorad heatmap: per-subspecialty diagnostic accuracy by model.

Shows accuracy values as annotated colored cells (green=high, red=low).

Usage:
    python figures/eurorad_heatmap.py
    python figures/eurorad_heatmap.py --output figures/eurorad_heatmap.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Model display order (columns)
MODEL_ORDER = [
    "GPT-5.1", "GPT-5-mini", "Gemini 3.1 Pro", "DeepSeek-R1",
    "gpt-oss-20b (H)", "gpt-oss-120b (H)",
    "Qwen3.5 9B", "Qwen3.5 27B", "Qwen3.5 35B", "Qwen3.5 35B FT",
]

MODEL_LABELS = {
    "GPT-5.1": "GPT-5.1",
    "GPT-5-mini": "GPT-5-mini",
    "Gemini 3.1 Pro": "Gemini 3.1",
    "DeepSeek-R1": "DeepSeek-R1",
    "gpt-oss-20b (H)": "gpt-oss-20b (H)",
    "gpt-oss-120b (H)": "gpt-oss-120b (H)",
    "Qwen3.5 9B": "Qwen3.5 9B",
    "Qwen3.5 27B": "Qwen3.5 27B",
    "Qwen3.5 35B": "Qwen3.5 35B",
    "Qwen3.5 35B FT": "Qwen3.5 35B FT",
}

# Row order (subspecialties)
CATEGORY_ORDER = [
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

CATEGORY_LABELS = {
    "Musculoskeletal system": "Musculoskeletal",
    "Cardiovascular": "Cardiovascular",
    "Abdominal imaging": "Abdominal",
    "Uroradiology & genital male imaging": "Uroradiology",
    "Neuroradiology": "Neuroradiology",
    "Paediatric radiology": "Paediatric",
    "Head & neck imaging": "Head & Neck",
    "Breast imaging": "Breast",
    "Chest imaging": "Chest",
    "Others": "Others",
}


def plot(output=None):
    df = pd.read_csv("statistics/all_statistics.csv")
    acc = df[(df["dataset"] == "Eurorad") & (df["analysis"] == "accuracy")]

    # Build matrix
    data = np.zeros((len(CATEGORY_ORDER), len(MODEL_ORDER)))
    for i, cat in enumerate(CATEGORY_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            row = acc[(acc["model_1"] == model) & (acc["category"] == cat)]
            if len(row):
                data[i, j] = row.iloc[0]["value"]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    # Colormap: red (low) → yellow (mid) → green (high)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "accuracy",
        ["#d62728", "#ffdd57", "#2ca02c"],
        N=256,
    )

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=40, vmax=100)

    # Annotate cells
    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(MODEL_ORDER)):
            val = data[i, j]
            color = "white" if val < 55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="medium", color=color)

    # Axes
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER],
                       fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
                       fontsize=10)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Anatomical Subgroup", fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Grid lines between cells
    for i in range(len(CATEGORY_ORDER) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(len(MODEL_ORDER) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    # Separator lines between model groups
    group_boundaries = [3.5, 5.5]  # after DeepSeek-R1, after gpt-oss-120b
    for gb in group_boundaries:
        ax.axvline(gb, color="#333333", linewidth=2.0)

    plt.tight_layout()

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
