#!/usr/bin/env python3
"""
Eurorad radar plot: per-subspecialty diagnostic accuracy by model.

Shows majority-vote accuracy across radiology subspecialties for selected
models, with dashed lines for proprietary references and solid lines for
on-device models.

Usage:
    python figures/eurorad_radar_plot.py
    python figures/eurorad_radar_plot.py --output figures/eurorad_radar.png
"""

import re
import argparse
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# ─── Models to include ──────────────────────────────────────────────────────

MODELS = {
    "GPT-5.1":       ["gpt-5.1-1113-M1", "gpt-5.1-1113-M2", "gpt-5.1-1113-M3"],
    "Gemini 3.1":    ["gemini-3.1-pro-M1", "gemini-3.1-pro-M2", "gemini-3.1-pro-M3"],
    "DeepSeek-R1":   ["deepseek r1 0528 v2", "deepseek r1 0528 v2.1", "deepseek r1 0528 v2.2"],
    "OSS-120B (H)":  ["oss-120b (H) v1", "oss-120b (H) v2", "oss-120b (H) v3"],
    "Qwen 35B (FT)": ["qwen3.5 35B fine-tuned v1", "qwen3.5 35B fine-tuned v2", "qwen3.5 35B fine-tuned v3"],
    "Gemma 4 31B":   ["gemma-4-31b-M1", "gemma-4-31b-M2", "gemma-4-31b-M3"],
}

MODEL_STYLES = {
    "GPT-5.1":       {"color": MODEL_COLORS["GPT-5.1"],       "ls": "--", "lw": 2.0, "fill_alpha": 0.08},
    "Gemini 3.1":    {"color": MODEL_COLORS["Gemini 3.1"],    "ls": "--", "lw": 2.0, "fill_alpha": 0.08},
    "DeepSeek-R1":   {"color": MODEL_COLORS["DeepSeek-R1"],   "ls": "--", "lw": 2.0, "fill_alpha": 0.08},
    "OSS-120B (H)":  {"color": MODEL_COLORS["OSS-120B (H)"],  "ls": "-",  "lw": 2.5, "fill_alpha": 0.10},
    "Qwen 35B (FT)": {"color": MODEL_COLORS["Qwen 35B (FT)"], "ls": "-",  "lw": 2.5, "fill_alpha": 0.15},
    "Gemma 4 31B":   {"color": MODEL_COLORS["Gemma 4 31B"],   "ls": "-",  "lw": 2.5, "fill_alpha": 0.10},
}

# ─── Subspecialty layout (clockwise from top, matching example) ─────────────

SECTION_ORDER = [
    "Musculoskeletal system",
    "Paediatric radiology",
    "Cardiovascular",
    "Abdominal imaging",
    "Uroradiology & genital male imaging",
    "Others",
    "Breast imaging",
    "Head & neck imaging",
    "Neuroradiology",
    "Chest imaging",
]

SECTION_LABELS = {
    "Musculoskeletal system":            "Musculoskeletal\nSystem",
    "Paediatric radiology":              "Paediatric\nRadiology",
    "Cardiovascular":                    "Cardiovascular",
    "Abdominal imaging":                 "Abdominal\nImaging",
    "Uroradiology & genital male imaging": "Uroradiology &\nGenital Male",
    "Others":                            "Others",
    "Breast imaging":                    "Breast Imaging",
    "Head & neck imaging":               "Head & Neck\nImaging",
    "Neuroradiology":                    "Neuroradiology",
    "Chest imaging":                     "Chest Imaging",
}

# Sections too small to plot individually → "Others"
OTHERS_SECTIONS = {"Genital (female) imaging", "Interventional radiology"}


# ─── Compute ────────────────────────────────────────────────────────────────

def compute_section_accuracy(df):
    """Per-section majority-vote accuracy for each model."""
    gt = df["FinalDiagnosis"].apply(norm_text)
    mask = gt.astype(bool)

    sections = df["Section"].copy()
    sections = sections.replace({s: "Others" for s in OTHERS_SECTIONS})

    results = {}
    for model_name, cols in MODELS.items():
        run_correct = []
        for col in cols:
            preds = df[col].apply(norm_text)
            run_correct.append((preds[mask] == gt[mask]).values)

        n = mask.sum()
        maj_correct = np.array([
            sum(run_correct[r][i] for r in range(len(cols))) >= 2
            for i in range(n)
        ])

        valid_sections = sections[mask].values
        section_accs = {}
        for sec in SECTION_ORDER:
            sec_mask = valid_sections == sec
            sec_total = sec_mask.sum()
            if sec_total > 0:
                section_accs[sec] = maj_correct[sec_mask].sum() / sec_total * 100
            else:
                section_accs[sec] = 0
        results[model_name] = section_accs

    return results


# ─── Plot ───────────────────────────────────────────────────────────────────

def plot(output=None):
    df = pd.read_csv("csvs/final_csvs/Eurorad.csv")
    df = df[df["FinalDiagnosis"].notna() &
            (df["FinalDiagnosis"].astype(str).str.strip() != "")]

    results = compute_section_accuracy(df)

    N = len(SECTION_ORDER)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Start from top (90°)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Radial axis
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"],
                       fontsize=8, color="grey")
    ax.yaxis.grid(True, color="grey", alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, color="grey", alpha=0.3, linewidth=0.5)

    # Angular axis — suppress default labels; we draw them manually after plotting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    # Plot each model
    for model_name in MODELS:
        accs = [results[model_name][sec] for sec in SECTION_ORDER]
        accs += accs[:1]

        style = MODEL_STYLES[model_name]
        ax.plot(angles, accs,
                linewidth=style["lw"], linestyle=style["ls"],
                color=style["color"], label=model_name,
                marker="o", markersize=4)
        ax.fill(angles, accs, alpha=style["fill_alpha"], color=style["color"])

    # Draw category labels on top of plot lines using polar data coords
    label_r = 115  # radial position beyond the 0-100 data range
    for angle, sec in zip(angles[:-1], SECTION_ORDER):
        # Determine horizontal alignment based on position around the circle
        cos_val = np.cos(angle - np.pi / 2)
        if cos_val > 0.1:
            ha = "left"
        elif cos_val < -0.1:
            ha = "right"
        else:
            ha = "center"
        ax.text(angle, label_r, SECTION_LABELS[sec],
                ha=ha, va="center", fontsize=10, fontweight="medium",
                bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.9),
                zorder=20)

    # Legend (horizontal, below the plot)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15),
              fontsize=9, framealpha=0.9, edgecolor="#cccccc",
              ncol=len(MODELS), columnspacing=1.0)

    # Spine
    ax.spines["polar"].set_color("grey")
    ax.spines["polar"].set_linewidth(0.5)

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
