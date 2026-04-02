#!/usr/bin/env python3
"""
NMED split violin plot: diagnosis vs treatment error distributions.

Shows per-model error (model score − human score) as split violins,
with diagnosis on one side and treatment on the other.

Usage:
    python figures/nmed_violin_plot.py
    python figures/nmed_violin_plot.py --output figures/nmed_violin.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection

from palette import MODEL_COLORS


# ─── Model definitions ──────────────────────────────────────────────────────
# (display_name, diagnosis_cols, treatment_cols)

MODELS = {
    "GPT-5-mini": {
        "diag": ["gpt-5-mini-0807-m1", "gpt-5-mini-0807-m2", "gpt-5-mini-0807-m3"],
        "treat": ["gpt-5-mini-0807-m1", "gpt-5-mini-0807-m2", "gpt-5-mini-0807-m3"],
    },
    "GPT-5.1": {
        "diag": ["gpt-5.1-1113-m1", "gpt-5.1-1113-m2", "gpt-5.1-1113-m3"],
        "treat": ["gpt-5.1-1113-m1", "gpt-5.1-1113-m2", "gpt-5.1-1113-m3"],
    },
    "Gemini 3.1": {
        "diag": ["gemini-3.1-pro-m1", "gemini-3.1-pro-m2", "gemini-3.1-pro-m3"],
        "treat": ["gemini-3.1-pro-m1", "gemini-3.1-pro-m2", "gemini-3.1-pro-m3"],
    },
    "DeepSeek-R1": {
        "diag": ["deepseek-0528-v1", "deepseek-0528-v2", "deepseek-0528-v3"],
        "treat": ["deepseek-r1-0528-v1", "deepseek-r1-0528-v2", "deepseek-r1-0528-v3"],
    },
    "OSS-20B (H)": {
        "diag": ["oss20b (H) v1", "oss20b (H) v2", "oss20b (H) v3"],
        "treat": ["oss20b (H) v1", "oss20b (H) v2", "oss20b (H) v3"],
    },
    "OSS-120B (H)": {
        "diag": ["oss120b (H) v1", "oss120b (H) v2", "oss120b (H) v3"],
        "treat": ["oss120b (H) v1", "oss120b (H) v2", "oss120b (H) v3"],
    },
    "Qwen 9B": {
        "diag": ["qwen3.5 9B v1", "qwen3.5 9B v2", "qwen3.5 9B v3"],
        "treat": ["qwen3.5 9B v1", "qwen3.5 9B v2", "qwen3.5 9B v3"],
    },
    "Qwen 35B": {
        "diag": ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
        "treat": ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
    },
}

SHORT_LABELS = {
    "GPT-5-mini": "GPT-5-mini",
    "GPT-5.1": "GPT-5.1",
    "Gemini 3.1": "Gemini 3.1",
    "DeepSeek-R1": "DeepSeek-R1",
    "OSS-20B (H)": "gpt-oss-20b (H)",
    "OSS-120B (H)": "gpt-oss-120b (H)",
    "Qwen 9B": "Qwen 9B",
    "Qwen 35B": "Qwen 35B",
}

# Split violin colors (matching example reference)
TREAT_COLOR = "#9DD5D0"   # pastel teal
DIAG_COLOR  = "#F5ADA6"   # pastel salmon/pink


def compute_errors(df, gt_col, model_cols):
    """Compute error = model_score - human_score, pooled across runs."""
    gt = pd.to_numeric(df[gt_col], errors="coerce")
    errors = []
    for col in model_cols:
        scores = pd.to_numeric(df[col], errors="coerce")
        mask = gt.notna() & scores.notna()
        errors.extend((scores[mask] - gt[mask]).tolist())
    return np.array(errors)


def plot(output=None):
    df_diag = pd.read_csv("csvs/NMED_Diagnosis.csv")
    df_treat = pd.read_csv("csvs/NMED_Treatment.csv")
    df_diag = df_diag[df_diag["Clinical specialty"].notna()]
    df_treat = df_treat[df_treat["Clinical specialty"].notna()]

    model_names = list(MODELS.keys())
    n_models = len(model_names)

    # Collect errors
    diag_errors = {}
    treat_errors = {}
    for name, cols in MODELS.items():
        diag_errors[name] = compute_errors(df_diag, "HumanEvalScore", cols["diag"])
        treat_errors[name] = compute_errors(df_treat, "HumanEvalScore", cols["treat"])

    fig, ax = plt.subplots(figsize=(16, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    positions = np.arange(n_models)

    # Draw split violins
    for i, name in enumerate(model_names):
        for side, data, color in [
            ("left", treat_errors[name], TREAT_COLOR),
            ("right", diag_errors[name], DIAG_COLOR),
        ]:
            if len(data) == 0:
                continue

            parts = ax.violinplot(data, positions=[i], showmeans=False,
                                  showmedians=False, showextrema=False,
                                  widths=0.8)

            for pc in parts["bodies"]:
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                if side == "left":
                    pc.get_paths()[0].vertices[:, 0] = np.clip(
                        pc.get_paths()[0].vertices[:, 0], -np.inf, m)
                else:
                    pc.get_paths()[0].vertices[:, 0] = np.clip(
                        pc.get_paths()[0].vertices[:, 0], m, np.inf)
                pc.set_facecolor(color)
                pc.set_edgecolor("grey")
                pc.set_linewidth(0.5)
                pc.set_alpha(0.7)

        # Two separate box plots: treatment (right edge at center),
        # diagnosis (left edge at center)
        box_w = 0.07
        gap = 0.015  # tiny gap between the two boxes
        _treat_outliers = np.array([])
        _diag_outliers = np.array([])
        for side, data in [("left", treat_errors[name]),
                           ("right", diag_errors[name])]:
            if len(data) == 0:
                continue

            q1, med, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1
            whisker_lo = max(data.min(), q1 - 1.5 * iqr)
            whisker_hi = min(data.max(), q3 + 1.5 * iqr)
            outliers = data[(data < whisker_lo) | (data > whisker_hi)]

            if side == "left":
                # Treatment: box right edge at center - gap, whisker at right edge
                box_left = i - gap - box_w
                wx = i - gap
            else:
                # Diagnosis: box left edge at center + gap, whisker at left edge
                box_left = i + gap
                wx = i + gap

            # Whiskers (solid)
            ax.plot([wx, wx], [whisker_lo, q1], color="black",
                    linewidth=0.8, zorder=3)
            ax.plot([wx, wx], [q3, whisker_hi], color="black",
                    linewidth=0.8, zorder=3)
            # Box
            rect = plt.Rectangle((box_left, q1), box_w, iqr,
                                  facecolor="white", edgecolor="black",
                                  linewidth=0.8, zorder=4)
            ax.add_patch(rect)
            # Median
            ax.plot([box_left, box_left + box_w], [med, med],
                    color="black", linewidth=1.2, zorder=5)
            # Collect outliers to draw combined after both sides
            if side == "left":
                _treat_outliers = outliers
            else:
                _diag_outliers = outliers

        # Combined outlier dots on center line
        all_outliers = np.concatenate([
            _treat_outliers if len(_treat_outliers) > 0 else np.array([]),
            _diag_outliers if len(_diag_outliers) > 0 else np.array([]),
        ])
        if len(all_outliers) > 0:
            ax.scatter([i] * len(all_outliers), all_outliers, color="black",
                       s=6, zorder=5, marker="o", linewidths=0.3)

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.8, zorder=1)

    # Axis formatting
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT_LABELS[m] for m in model_names],
                       fontsize=10, rotation=0)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_ylim(-4.0, 4.0)
    ax.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", length=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")

    # Legend
    treat_patch = mpatches.Patch(color=TREAT_COLOR, alpha=0.7, label="treatment")
    diag_patch = mpatches.Patch(color=DIAG_COLOR, alpha=0.7, label="diagnosis")
    ax.legend(handles=[treat_patch, diag_patch], loc="upper right",
              fontsize=10, framealpha=0.9, edgecolor="#cccccc")

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
