#!/usr/bin/env python3
"""
Supplementary forest plots comparing Qwen3.5-35B FT against reference models.

Generates three plots:
  1. Qwen3.5-35B FT vs DeepSeek-R1
  2. Qwen3.5-35B FT vs GPT-5-mini
  3. Qwen3.5-35B FT vs GPT-5.1

Usage:
    python figures/forest_plot.py
    python figures/forest_plot.py --output-dir paper/supp-imgs
"""

import argparse
import csv
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import MODEL_COLORS


OTHERS_SECTIONS = {"Genital (female) imaging", "Interventional radiology"}

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

DISPLAY_NAMES = {
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

MODELS = {
    "GPT-5.1":        ["gpt-5.1-1113-M1", "gpt-5.1-1113-M2", "gpt-5.1-1113-M3"],
    "GPT-5-mini":     ["gpt-5-mini-0807-M1", "gpt-5-mini-0807-M2", "gpt-5-mini-0807-M3"],
    "DeepSeek-R1":    ["deepseek r1 0528 v2", "deepseek r1 0528 v2.1", "deepseek r1 0528 v2.2"],
    "Qwen3.5 35B FT": ["qwen3.5 35B fine-tuned v1", "qwen3.5 35B fine-tuned v2", "qwen3.5 35B fine-tuned v3"],
}

PLOT_CONFIGS = [
    {
        "reference": "DeepSeek-R1",
        "ref_color": "#999999",
        "ft_color": "#5E9CD0",
        "filename": "manuscript_forest_plot_deepseek.png",
    },
    {
        "reference": "GPT-5-mini",
        "ref_color": "#999999",
        "ft_color": "#E06666",
        "filename": "manuscript_forest_plot_gpt5mini.png",
    },
    {
        "reference": "GPT-5.1",
        "ref_color": "#999999",
        "ft_color": "#6AA84F",
        "filename": "manuscript_forest_plot_gpt5.png",
    },
]


def norm_text(s):
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
    return p_hat * 100, max(0.0, centre - margin) * 100, min(1.0, centre + margin) * 100


def load_data():
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
    df = df[df["FinalDiagnosis"].apply(lambda x: bool(str(x).strip()))].copy()
    return df


def compute_results(df):
    gt = df["FinalDiagnosis"].apply(norm_text).values
    sections = df["Section"].values
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

        majority = [sum(run_correct[r][i] for r in range(len(cols))) >= 2
                     for i in range(len(gt))]

        model_results = {}
        k_total = sum(majority)
        n_total = len(majority)
        acc, lo, hi = wilson_ci(k_total, n_total)
        model_results["Overall"] = (acc, lo, hi)

        for cat in CATEGORIES:
            mask = [s == cat for s in sections]
            n_cat = sum(mask)
            k_cat = sum(c for c, m in zip(majority, mask) if m)
            acc, lo, hi = wilson_ci(k_cat, n_cat)
            model_results[cat] = (acc, lo, hi)

        results[model_name] = model_results

    return results


def draw_forest_plot(results, config, output_path=None):
    ref_name = config["reference"]
    ft_name = "Qwen3.5 35B FT"
    ref_color = config["ref_color"]
    ft_color = config["ft_color"]

    ref_data = results[ref_name]
    ft_data = results[ft_name]

    categories = list(reversed(CATEGORIES))
    n_cats = len(categories)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(n_cats) * 2.0
    offset = 0.35

    ref_overall = ref_data["Overall"][0]
    ft_overall = ft_data["Overall"][0]

    ax.axvline(ref_overall, color=ref_color, linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
    ax.axvline(ft_overall, color=ft_color, linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

    for i, cat in enumerate(categories):
        y = y_positions[i]

        ref_acc, ref_lo, ref_hi = ref_data[cat]
        ft_acc, ft_lo, ft_hi = ft_data[cat]

        ax.plot([ref_lo, ref_hi], [y - offset, y - offset],
                color=ref_color, linewidth=1.5, zorder=2)
        ax.plot(ref_acc, y - offset, 'o', color=ref_color, markersize=6, zorder=3)

        ax.plot([ft_lo, ft_hi], [y + offset, y + offset],
                color=ft_color, linewidth=1.5, zorder=2)
        ax.plot(ft_acc, y + offset, 'o', color=ft_color, markersize=6, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([DISPLAY_NAMES[c] for c in categories], fontsize=11, fontweight="medium")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 105)

    ax.grid(axis="x", alpha=0.2, color="#cccccc", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)

    ref_label = f"{ref_name} Overall ({ref_overall:.1f}%)"
    ft_label = f"Fine-tuned Qwen3.5-35B Overall ({ft_overall:.1f}%)"

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=ref_color, linestyle="--", linewidth=0.8,
               marker="o", markersize=6, label=ref_label),
        Line2D([0], [0], color=ft_color, linestyle="--", linewidth=0.8,
               marker="o", markersize=6, label=ft_label),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              frameon=True, fancybox=False, edgecolor="#cccccc")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", "-o", default=None,
                    help="Directory to save plots (default: show)")
    args = ap.parse_args()

    df = load_data()
    results = compute_results(df)

    for config in PLOT_CONFIGS:
        output_path = None
        if args.output_dir:
            output_path = str(Path(args.output_dir) / config["filename"])
        draw_forest_plot(results, config, output_path)


if __name__ == "__main__":
    main()
