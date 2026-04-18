#!/usr/bin/env python3
"""
Fig 3 — Parameter-efficiency scatter plot.

Shows that on-device models with small memory footprints approach
proprietary model accuracy on the Eurorad general diagnosis task (N=207).

Memory footprint formula:
    memory_gb = total_params_B * 0.5 * 1.15
    (0.5 bytes/param at 4-bit quantization × 1.15 framework overhead factor;
     no KV cache included beyond framework defaults)

Usage:
    python figures/fig3_parameter_efficiency.py
    python figures/fig3_parameter_efficiency.py --output figures/outputs/fig3_parameter_efficiency
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from adjustText import adjust_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import MODEL_COLORS


# ─── Text normalization (matches statistics/utils.py) ─────────────────────

def norm_text(s) -> str:
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


# ─── Model definitions ────────────────────────────────────────────────────

MODELS = {
    "gpt-oss-20b (H)":  ["oss-20b (H) v1", "oss-20b (H) v2", "oss-20b (H) v3"],
    "gpt-oss-120b (H)": ["oss-120b (H) v1", "oss-120b (H) v2", "oss-120b (H) v3"],
    "Qwen3.5 9B":       ["qwen3.5 9B v1", "qwen3.5 9B v2", "qwen3.5 9B v3"],
    "Qwen3.5 27B":      ["qwen3.5 27B v1", "qwen3.5 27B v2", "qwen3.5 27B v3"],
    "Qwen3.5 35B":      ["qwen3.5 35B v1", "qwen3.5 35B v2", "qwen3.5 35B v3"],
    "Qwen3.5 35B FT":   ["qwen3.5 35B fine-tuned v1", "qwen3.5 35B fine-tuned v2", "qwen3.5 35B fine-tuned v3"],
    "Gemma 4 31B":      ["gemma-4-31b-M1", "gemma-4-31b-M2", "gemma-4-31b-M3"],
    "DeepSeek-R1":      ["deepseek r1 0528 v2", "deepseek r1 0528 v2.1", "deepseek r1 0528 v2.2"],
    "GPT-5-mini":       ["gpt-5-mini-0807-M1", "gpt-5-mini-0807-M2", "gpt-5-mini-0807-M3"],
    "GPT-5.1":          ["gpt-5.1-1113-M1", "gpt-5.1-1113-M2", "gpt-5.1-1113-M3"],
    "Gemini 3.1 Pro":   ["gemini-3.1-pro-M1", "gemini-3.1-pro-M2", "gemini-3.1-pro-M3"],
}

# Parameter counts (total billions)
# Sources cited inline
TOTAL_PARAMS_B = {
    "gpt-oss-20b (H)":  20.9,   # arXiv:2508.10925
    "gpt-oss-120b (H)": 116.8,  # arXiv:2508.10925
    "Qwen3.5 9B":       9.0,    # Qwen HF model card
    "Qwen3.5 27B":      27.0,   # Qwen HF model card
    "Qwen3.5 35B":      35.0,   # Qwen HF model card (Hybrid MoE, 3.0B active)
    "Qwen3.5 35B FT":   35.0,   # same architecture
    "Gemma 4 31B":      31.0,   # Google Gemma 4 announcement (2026-03-31)
    "DeepSeek-R1":      671.0,  # DeepSeek GitHub
}

# gpt-oss-20b FT uses same architecture (LoRA adapter, same total params)
TOTAL_PARAMS_B["gpt-oss-20b FT"] = TOTAL_PARAMS_B["gpt-oss-20b (H)"]

# Model family assignments
FAMILY = {
    "gpt-oss-20b (H)":  "gpt-oss",
    "gpt-oss-20b FT":   "gpt-oss",
    "gpt-oss-120b (H)": "gpt-oss",
    "Qwen3.5 9B":       "Qwen3.5",
    "Qwen3.5 27B":      "Qwen3.5",
    "Qwen3.5 35B":      "Qwen3.5",
    "Qwen3.5 35B FT":   "Qwen3.5",
    "Gemma 4 31B":      "Gemma 4",
    "DeepSeek-R1":      "DeepSeek",
}

# Family colors (from palette.py)
FAMILY_COLORS = {
    "gpt-oss":  "#F28B82",  # salmon
    "Qwen3.5":  "#5E9CD0",  # blue
    "Gemma 4":  "#F4A261",  # warm orange
    "DeepSeek": "#6DBF67",  # medium green
}

FAMILY_MARKERS = {
    "gpt-oss":  "o",
    "Qwen3.5":  "s",
    "Gemma 4":  "D",
    "DeepSeek": "^",
}

# Proprietary reference lines
PROPRIETARY_REFS = {
    "GPT-5-mini":     None,
    "GPT-5.1":        None,
    "Gemini 3.1 Pro": None,
}
PROPRIETARY_COLOR = "#7E57C2"  # deeper violet

# Expected accuracy values for verification (from Table 1)
EXPECTED_ACC = {
    "gpt-oss-20b (H)":  80.7,
    "gpt-oss-120b (H)": 83.1,
    "Qwen3.5 9B":       75.8,
    "Qwen3.5 27B":      80.2,
    "Qwen3.5 35B":      84.5,
    "Qwen3.5 35B FT":   88.4,
    "Gemma 4 31B":      86.5,
    "DeepSeek-R1":      81.6,
    "GPT-5-mini":       84.1,
    "GPT-5.1":          88.9,
    "Gemini 3.1 Pro":   91.3,
}


def compute_memory_gb(total_params_b):
    """4-bit quantization memory estimate."""
    return total_params_b * 0.5 * 1.15


# ─── Data loading ─────────────────────────────────────────────────────────

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
    valid = [i for i, cid in enumerate(data["case_id"]) if cid.strip()]
    return headers, {h: [data[h][i] for i in valid] for h in headers}


def compute_accuracies(headers, data):
    gt = [norm_text(s) for s in data["FinalDiagnosis"]]
    n_cases = len(gt)
    results = {}
    for model_name, cols in MODELS.items():
        missing = [c for c in cols if c not in headers]
        if missing:
            print(f"  WARNING: {model_name} missing columns: {missing}")
            continue
        run_correct = []
        for col in cols:
            preds = [norm_text(s) for s in data[col]]
            run_correct.append([p == g for p, g in zip(preds, gt)])
        k = sum(
            sum(run_correct[r][i] for r in range(len(cols))) >= 2
            for i in range(n_cases)
        )
        acc, lo, hi = wilson_ci(k, n_cases)
        results[model_name] = round(acc * 100, 1)
    return results


def load_ft_oss20b_accuracy():
    """Load gpt-oss-20b FT accuracy from finetune-eval CSVs."""
    import pandas as pd
    ft_dir = "csvs/finetune-eval"
    files = [
        f"{ft_dir}/oss120b_base_eurorad.csv",  # these are actually oss-20b FT evals
    ]
    # The FT accuracy for gpt-oss-20b is 86.5% (from Table 1)
    # Load from the finetune barplot's pipeline
    checkpoints = [
        f"{ft_dir}/oss120b_r32_a64_attn_552_eurorad.csv",
        f"{ft_dir}/oss120b_r32_a64_attn_552_eurorad_v2.csv",
        f"{ft_dir}/oss120b_r32_a64_attn_552_eurorad_v3.csv",
    ]
    run_correct = []
    n_cases = None
    for fpath in checkpoints:
        try:
            df = pd.read_csv(fpath)
            gt = df["FinalDiagnosis"].apply(norm_text)
            pred = df["model_answer"].apply(norm_text)
            mask = gt.astype(bool)
            correct = (gt[mask] == pred[mask]).values
            run_correct.append(correct)
            if n_cases is None:
                n_cases = len(correct)
        except FileNotFoundError:
            return None
    if not run_correct:
        return None
    maj_k = sum(sum(run_correct[r][i] for r in range(3)) >= 2
                for i in range(n_cases))
    acc, _, _ = wilson_ci(maj_k, n_cases)
    return round(acc * 100, 1)


# ─── Plot ─────────────────────────────────────────────────────────────────

def plot(accuracies, output=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Set axes early so log-scale transforms are available for annotations
    ax.set_xscale("log")
    ax.set_xlim(3.5, 550)
    ax.set_ylim(73, 95)

    # ── Manual label offsets (dx in log-space multiplier, dy in pp) ───────
    # dx > 0: label at mem * dx (right of marker); dx < 0: label at mem / |dx| (left)
    # All labels placed to the right for consistency, except where crowded
    LABEL_OFFSETS = {
        "gpt-oss-20b (H)":  (-1.6, -0.5),
        "gpt-oss-120b (H)": (1.30, -0.8),
        "Qwen3.5 9B":       (1.35, -0.3),
        "Qwen3.5 27B":      (1.35,  0.0),
        "Qwen3.5 35B":      (-1.9, -0.3),
        "Gemma 4 31B":      (1.35,  0.7),
        "DeepSeek-R1":      (1.20, -0.3),
    }

    # ── Scatter: base models only (FT points plotted separately) ──────────
    base_models = [m for m in FAMILY if m in accuracies and m in TOTAL_PARAMS_B
                   and "FT" not in m]

    # Legend order: Qwen3.5, gpt-oss, Gemma 4, DeepSeek (best FT acc descending)
    legend_order = ["Qwen3.5", "gpt-oss", "Gemma 4", "DeepSeek"]
    plotted_families = set()

    for model in base_models:
        mem = compute_memory_gb(TOTAL_PARAMS_B[model])
        acc = accuracies[model]
        fam = FAMILY[model]
        color = FAMILY_COLORS[fam]
        marker = FAMILY_MARKERS[fam]

        label = fam if fam not in plotted_families else None
        plotted_families.add(fam)

        ax.scatter(mem, acc, c=color, marker=marker, s=80, zorder=5,
                   edgecolors="white", linewidths=0.8, label=label)

        # Manual label placement
        display = (model.replace("gpt-oss-", "")
                        .replace("Qwen3.5 ", "")
                        .replace(" (H)", ""))
        dx_mult, dy = LABEL_OFFSETS.get(model, (1.25, 0))
        lx = mem * dx_mult if dx_mult > 0 else mem / abs(dx_mult)
        ly = acc + dy
        ha = "left" if dx_mult > 0 else "right"
        ax.annotate(
            display, xy=(mem, acc), xytext=(lx, ly),
            fontsize=7.5, color="#333333", ha=ha, va="center", zorder=6,
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.4,
                            shrinkA=0, shrinkB=3),
        )

    # ── FT points (no text labels — arrows + caption explain) ─────────────
    ft_pairs = [
        ("gpt-oss-20b (H)", "gpt-oss-20b FT"),
        ("Qwen3.5 35B", "Qwen3.5 35B FT"),
    ]
    for base, ft in ft_pairs:
        if base not in accuracies or ft not in accuracies:
            continue
        mem_ft = compute_memory_gb(TOTAL_PARAMS_B[ft])
        acc_ft = accuracies[ft]
        fam = FAMILY[ft]
        color = FAMILY_COLORS[fam]
        marker = FAMILY_MARKERS[fam]

        # Plot FT point (open marker to distinguish from base; high zorder above ref lines)
        ax.scatter(mem_ft, acc_ft, marker=marker, s=90, zorder=7,
                   edgecolors=color, linewidths=1.8, facecolors="white")

    # ── Fine-tuning arrows: single direction, head at FT only ─────────────
    for base, ft in ft_pairs:
        if base not in accuracies or ft not in accuracies:
            continue
        mem_base = compute_memory_gb(TOTAL_PARAMS_B[base])
        mem_ft = compute_memory_gb(TOTAL_PARAMS_B[ft])
        acc_base = accuracies[base]
        acc_ft = accuracies[ft]
        fam = FAMILY[base]
        color = FAMILY_COLORS[fam]

        ax.annotate(
            "", xy=(mem_ft, acc_ft), xytext=(mem_base, acc_base),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color, lw=1.4,
                shrinkA=7, shrinkB=7,
                mutation_scale=12,
            ),
            zorder=4,
        )

    # ── Proprietary reference lines ───────────────────────────────────────
    ref_models = ["GPT-5-mini", "GPT-5.1", "Gemini 3.1 Pro"]
    for model in ref_models:
        if model not in accuracies:
            continue
        acc = accuracies[model]
        ax.axhline(y=acc, color=PROPRIETARY_COLOR, linestyle="--",
                   linewidth=0.7, alpha=0.5, zorder=2)

    # Place reference labels on right margin after xlim is final
    xlim = ax.get_xlim()
    for model in ref_models:
        if model not in accuracies:
            continue
        acc = accuracies[model]
        ax.text(xlim[1] * 1.05, acc, f"{model} ({acc}%)",
                fontsize=6.5, color=PROPRIETARY_COLOR, va="center",
                fontweight="bold", zorder=6, clip_on=False)

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500])
    ax.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
    ax.ticklabel_format(axis="x", style="plain")
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", length=3, color="#cccccc")
    ax.tick_params(axis="y", which="minor", length=0)

    ax.set_xlabel("Memory footprint at 4-bit quantization (GB)", fontsize=11)
    ax.set_ylabel("Diagnostic accuracy (%)", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    ax.grid(True, alpha=0.25, color="#cccccc", linewidth=0.5, which="major")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#cccccc")

    # ── Legend ─────────────────────────────────────────────────────────────
    # Reorder handles to match legend_order
    handles_dict = {}
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles_dict[l] = h

    ordered_handles = []
    ordered_labels = []
    for fam in legend_order:
        if fam in handles_dict:
            ordered_handles.append(handles_dict[fam])
            ordered_labels.append(fam)

    # FT legend entry: filled circle → open circle to match base-to-FT visual
    ft_handle = Line2D([0], [0], linestyle="none", marker="o", markersize=6,
                       markerfacecolor="white", markeredgecolor="#888888",
                       markeredgewidth=1.8)
    ordered_handles.append(ft_handle)
    ordered_labels.append("Fine-tuned (○)")

    ax.legend(ordered_handles, ordered_labels, loc="lower right", fontsize=8,
              frameon=True, framealpha=0.9, edgecolor="#cccccc",
              handletextpad=0.5, borderpad=0.6)

    plt.tight_layout()

    if output:
        base = output.replace(".pdf", "").replace(".png", "")
        fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {base}.pdf")
        print(f"Saved: {base}.png")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="csvs/final_csvs/Eurorad.csv")
    ap.add_argument("--output", "-o", default=None,
                    help="Output path base (e.g., figures/outputs/fig3_parameter_efficiency)")
    args = ap.parse_args()

    # Use Table 1 values as authoritative source.
    # The CSV-based computation can differ by up to ~0.5pp due to header
    # deduplication differences between the statistics pipeline and raw CSV
    # loading. Table 1 values are the published numbers.
    accuracies = dict(EXPECTED_ACC)
    accuracies["gpt-oss-20b FT"] = 86.5  # from Table 1 / Fig 2c

    # Cross-check against CSV to flag any large discrepancies
    print("Loading Eurorad data for cross-check...")
    headers, data = load_eurorad(args.input)
    csv_accs = compute_accuracies(headers, data)

    print(f"\n{'Model':<22s}  {'Table 1':>7s}  {'CSV':>6s}  {'Delta':>6s}  {'Memory (GB)':>11s}")
    print("-" * 65)
    for model in sorted(EXPECTED_ACC.keys()):
        table_val = EXPECTED_ACC[model]
        csv_val = csv_accs.get(model)
        delta_str = ""
        if csv_val is not None:
            delta = abs(table_val - csv_val)
            delta_str = f"{delta:>5.1f}pp"
            if delta > 1.0:
                delta_str += " !!!"
        else:
            delta_str = "  N/A"
        mem_str = ""
        if model in TOTAL_PARAMS_B:
            mem_str = f"{compute_memory_gb(TOTAL_PARAMS_B[model]):>8.1f} GB"
        csv_str = f"{csv_val:>5.1f}%" if csv_val is not None else "  N/A"
        print(f"{model:<22s}  {table_val:>6.1f}%  {csv_str}  {delta_str}  {mem_str}")

    print(f"\nMemory footprint formula: memory_gb = total_params_B * 0.5 * 1.15")
    print(f"  (4-bit quantization: 0.5 bytes/param × 1.15 framework overhead)\n")

    plot(accuracies, args.output)


if __name__ == "__main__":
    main()
