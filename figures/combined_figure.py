#!/usr/bin/env python3
"""
Combined multi-panel figure (a–d) for the paper.

  a) Eurorad accuracy bar plot         (full width)
  b) NMED split violin plot            (full width)
  c) Eurorad fine-tune bar plot        (bottom-left)
  d) Eurorad radar plot                (bottom-right)

Usage:
    python figures/combined_figure.py
    python figures/combined_figure.py --output figures/combined.png
"""

import sys, os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Allow imports from figures/
sys.path.insert(0, os.path.dirname(__file__))
from palette import MODEL_COLORS

# Reuse data-loading helpers from individual scripts
from eurorad_accuracy_barplot import (
    load_eurorad, compute_accuracy, wilson_ci,
    MODELS as ACC_MODELS, MODEL_ORDER as ACC_ORDER,
    SHORT_LABELS as ACC_SHORT, GROUPS as ACC_GROUPS,
    norm_text,
)
from eurorad_finetune_barplot import (
    load_eurorad_results, load_finetune_eval_results,
    draw_bracket,
)
from eurorad_radar_plot import (
    compute_section_accuracy,
    MODELS as RADAR_MODELS, MODEL_STYLES,
    SECTION_ORDER, SECTION_LABELS,
)
from nmed_violin_plot import (
    compute_errors,
    MODELS as NMED_MODELS, SHORT_LABELS as NMED_SHORT,
    TREAT_COLOR, DIAG_COLOR,
)


# ─── Panel a: Eurorad accuracy bar plot ────────────────────────────────────

def panel_accuracy(ax):
    headers, data = load_eurorad()
    results = compute_accuracy(headers, data)

    positions, colors, labels = [], [], []
    step = 1.6
    gap = 1.8
    x = 0
    group_spans = {}

    for group_name, model_names in ACC_GROUPS.items():
        start = x
        for m in model_names:
            if m in results:
                positions.append(x)
                colors.append(MODEL_COLORS[m])
                labels.append(m)
                x += step
        group_spans[group_name] = (start, x - step)
        x += gap

    accs = [results[m][0] * 100 for m in labels]
    los = [results[m][1] * 100 for m in labels]
    his = [results[m][2] * 100 for m in labels]
    err_lo = [a - l for a, l in zip(accs, los)]
    err_hi = [h - a for a, h in zip(accs, his)]

    ax.bar(positions, accs, yerr=[err_lo, err_hi], capsize=4,
           color=colors, edgecolor="none", width=0.72,
           error_kw={"linewidth": 1.2, "color": "black", "capthick": 1.2})

    for pos, acc in zip(positions, accs):
        ax.text(pos, acc + 1.5, f"{acc:.1f}", ha="center", va="center",
                fontsize=7, color="#333333", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))

    trans = ax.get_xaxis_transform()
    line_y = -0.08
    text_y = -0.12
    for group_name, (start, end) in group_spans.items():
        mid = (start + end) / 2
        # Always draw a bracket line (even for single-bar groups like Gemma 4)
        ax.plot([start - 0.35, end + 0.35], [line_y, line_y],
                color="#999999", lw=0.8, transform=trans, clip_on=False)
        ax.text(mid, text_y, group_name, ha="center", va="top",
                fontsize=11, fontweight="bold", color="#444444",
                transform=trans)

    ax.set_xticks(positions)
    ax.set_xticklabels([ACC_SHORT[m] for m in labels], fontsize=8)
    ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=10)
    ax.set_ylim(60, 95)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, color="#cccccc", linewidth=0.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ─── Panel b: NMED split violin plot ──────────────────────────────────────

def panel_violin(ax):
    df_diag = pd.read_csv("csvs/final_csvs/NMED_Diagnosis.csv")
    df_treat = pd.read_csv("csvs/final_csvs/NMED_Treatment.csv")
    df_diag = df_diag[df_diag["Clinical specialty"].notna()]
    df_treat = df_treat[df_treat["Clinical specialty"].notna()]

    model_names = list(NMED_MODELS.keys())
    n_models = len(model_names)

    diag_errors, treat_errors = {}, {}
    for name, cols in NMED_MODELS.items():
        diag_errors[name] = compute_errors(df_diag, "HumanEvalScore", cols["diag"])
        treat_errors[name] = compute_errors(df_treat, "HumanEvalScore", cols["treat"])

    positions = np.arange(n_models)

    for i, name in enumerate(model_names):
        for side, data, color in [
            ("left", treat_errors[name], TREAT_COLOR),
            ("right", diag_errors[name], DIAG_COLOR),
        ]:
            if len(data) == 0:
                continue
            parts = ax.violinplot(data, positions=[i], showmeans=False,
                                  showmedians=False, showextrema=False, widths=0.8)
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

        # Box plots
        box_w = 0.07
        gap = 0.015
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
                box_left = i - gap - box_w
                wx = i - gap
                _treat_outliers = outliers
            else:
                box_left = i + gap
                wx = i + gap
                _diag_outliers = outliers

            ax.plot([wx, wx], [whisker_lo, q1], color="black", linewidth=0.8, zorder=3)
            ax.plot([wx, wx], [q3, whisker_hi], color="black", linewidth=0.8, zorder=3)
            rect = plt.Rectangle((box_left, q1), box_w, iqr,
                                  facecolor="white", edgecolor="black",
                                  linewidth=0.8, zorder=4)
            ax.add_patch(rect)
            ax.plot([box_left, box_left + box_w], [med, med],
                    color="black", linewidth=1.2, zorder=5)

        all_outliers = np.concatenate([
            _treat_outliers if len(_treat_outliers) > 0 else np.array([]),
            _diag_outliers if len(_diag_outliers) > 0 else np.array([]),
        ])
        if len(all_outliers) > 0:
            ax.scatter([i] * len(all_outliers), all_outliers, color="black",
                       s=4, zorder=5, marker="o", linewidths=0.2)

    ax.axhline(y=0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([NMED_SHORT[m] for m in model_names], fontsize=8)
    ax.set_ylabel("Error", fontsize=10)
    ax.set_ylim(-4.0, 4.0)
    ax.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", length=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")

    treat_patch = mpatches.Patch(color=TREAT_COLOR, alpha=0.7, label="treatment")
    diag_patch = mpatches.Patch(color=DIAG_COLOR, alpha=0.7, label="diagnosis")
    ax.legend(handles=[treat_patch, diag_patch], loc="upper right",
              fontsize=8, framealpha=0.9, edgecolor="#cccccc")


# ─── Panel c: Eurorad finetune bar plot ───────────────────────────────────

def panel_finetune(ax):
    euro_results = load_eurorad_results()
    ft_results = load_finetune_eval_results()

    bar_names = [
        "DeepSeek", "GPT-5-mini", "GPT-5.1", "Gemini 3.1",
        "Qwen 35B", "35B (FT)",
    ]
    bars = [(name, MODEL_COLORS[name]) for name in bar_names]
    all_results = {**euro_results, **ft_results}

    positions = []
    x = 0
    step = 1.25
    gaps_after = {3}
    for idx in range(len(bars)):
        positions.append(x)
        x += step
        if idx in gaps_after:
            x += 0.9

    accs, los, his, colors, labels = [], [], [], [], []
    for label, color in bars:
        acc, lo, hi = all_results[label]
        accs.append(acc); los.append(lo); his.append(hi)
        colors.append(color); labels.append(label)

    err_lo = [a - l for a, l in zip(accs, los)]
    err_hi = [h - a for a, h in zip(accs, his)]

    ax.bar(positions, accs, yerr=[err_lo, err_hi], capsize=3,
           color=colors, edgecolor="none", width=0.72,
           error_kw={"linewidth": 1.0, "color": "black", "capthick": 1.0})

    for pos, acc in zip(positions, accs):
        ax.text(pos, acc + 1.5, f"{acc:.1f}", ha="center", va="center",
                fontsize=7, color="#333333", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))

    ft_pairs = [(4, 5)]
    for base_idx, ft_idx in ft_pairs:
        delta = accs[ft_idx] - accs[base_idx]
        bracket_y = max(his[base_idx], his[ft_idx]) + 0.5
        draw_bracket(ax, positions[base_idx], positions[ft_idx],
                     bracket_y, f"+{delta:.1f}%")

    trans = ax.get_xaxis_transform()
    groups = [("Proprietary", 0, 3), ("Qwen 3.5", 4, 5)]
    line_y = -0.08
    text_y = -0.12
    for group_name, i_start, i_end in groups:
        mid = (positions[i_start] + positions[i_end]) / 2
        ax.plot([positions[i_start] - 0.35, positions[i_end] + 0.35],
                [line_y, line_y], color="#999999", lw=0.8,
                transform=trans, clip_on=False)
        ax.text(mid, text_y, group_name, ha="center", va="top",
                fontsize=11, fontweight="bold", color="#444444",
                transform=trans)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=10)
    ax.set_ylim(60, 100)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, color="#cccccc", linewidth=0.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ─── Panel d: Eurorad radar plot ──────────────────────────────────────────

def panel_radar(ax):
    df = pd.read_csv("csvs/final_csvs/Eurorad.csv")
    df = df[df["FinalDiagnosis"].notna() &
            (df["FinalDiagnosis"].astype(str).str.strip() != "")]
    results = compute_section_accuracy(df)

    N = len(SECTION_ORDER)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="grey")
    ax.yaxis.grid(True, color="grey", alpha=0.3, linewidth=0.5)
    ax.xaxis.grid(True, color="grey", alpha=0.3, linewidth=0.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # suppress default labels; we'll draw them manually

    for model_name in RADAR_MODELS:
        accs = [results[model_name][sec] for sec in SECTION_ORDER]
        accs += accs[:1]
        style = MODEL_STYLES[model_name]
        ax.plot(angles, accs, linewidth=style["lw"], linestyle=style["ls"],
                color=style["color"], label=model_name, marker="o", markersize=3)
        ax.fill(angles, accs, alpha=style["fill_alpha"], color=style["color"])

    # Draw category labels on top of plot lines using polar data coords
    label_r = 115
    for angle, sec in zip(angles[:-1], SECTION_ORDER):
        cos_val = np.cos(angle - np.pi / 2)
        if cos_val > 0.1:
            ha = "left"
        elif cos_val < -0.1:
            ha = "right"
        else:
            ha = "center"
        ax.text(angle, label_r, SECTION_LABELS[sec],
                ha=ha, va="center", fontsize=8, fontweight="medium",
                bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.9),
                zorder=20)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15),
              fontsize=7, framealpha=0.9, edgecolor="#cccccc",
              ncol=len(RADAR_MODELS), columnspacing=1.0)
    ax.spines["polar"].set_color("grey")
    ax.spines["polar"].set_linewidth(0.5)


# ─── Combined figure ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", default=None)
    args = ap.parse_args()

    fig = plt.figure(figsize=(14, 20))
    fig.patch.set_facecolor("white")

    # GridSpec: 3 rows — full-width a, full-width b, split c+d
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1, 1, 1.4],
                           width_ratios=[1.1, 1.1],
                           hspace=0.35, wspace=0.35)

    # Panel a — full width
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_facecolor("white")
    panel_accuracy(ax_a)

    # Panel b — full width
    ax_b = fig.add_subplot(gs[1, :])
    ax_b.set_facecolor("white")
    panel_violin(ax_b)

    # Panel c — bottom left
    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.set_facecolor("white")
    panel_finetune(ax_c)

    # Panel d — bottom right (polar)
    ax_d = fig.add_subplot(gs[2, 1], polar=True)
    ax_d.set_facecolor("white")
    panel_radar(ax_d)

    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
