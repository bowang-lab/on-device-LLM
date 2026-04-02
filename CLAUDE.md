# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**on-device-LLM**: Benchmarks on-device large language models (gpt-oss and Qwen3.5 families) on medical diagnostic tasks across three clinical benchmarks.

### Primary Use Case
Evaluating on-device LLMs for clinical decision support:
1. **Eurorad** - Radiology case diagnosis (multiple-choice selection, N=207)
2. **NMED** - Diagnosis and treatment scoring using Likert rubrics (1-5 scale)
3. **Ophthalmology** - Eye disease diagnostic classification (MCQ)

### Model Lineup
**Proprietary references:** GPT-5.1, GPT-5-mini, Gemini 3.1 Pro
**Open-source:** DeepSeek-R1
**On-device gpt-oss:** gpt-oss-20b (H), gpt-oss-120b (H)
**On-device Qwen3.5:** 9B, 27B, 35B (base), 35B (fine-tuned)

### Excluded Models
GPT-5 (0807), GPT-5.2 (1211), and o4-mini are excluded from all figures and analyses.
They should not appear in plots, tables, or statistical comparisons.

## High-Level Architecture

```
on-device-LLM/
├── benchmarks/           # Evaluation scripts for each domain
│   ├── eurorad/          # Radiology case evaluation (8 scripts)
│   │   ├── hf_bench.py           # HF Inference Provider API
│   │   ├── gpt.py                # OpenAI API (GPT-5.1)
│   │   ├── openrouter.py         # OpenRouter API (DeepSeek, Qwen)
│   │   ├── eurorad_beams_hf.py   # Local: diverse beam search (HF)
│   │   ├── eurorad_beams_unsloth.py  # Local: diverse beam search (Unsloth)
│   │   ├── cot_beams_score_hf.py # CoT beams with log-prob rescoring
│   │   ├── eval_finetune.py      # LoRA checkpoint evaluation
│   │   └── oss20b_inference.py   # OSS20B local inference
│   ├── nmed-notes/       # NMED Likert scoring (6 scripts)
│   │   ├── hf_bench.py           # HF Inference Provider API
│   │   ├── gpt.py                # OpenAI API (GPT-5.1)
│   │   ├── openrouter.py         # OpenRouter API
│   │   ├── beams_hf.py           # Beam search (--task diagnosis/treatment)
│   │   ├── nmed_beams_local.py   # Local beam search (--task flag)
│   │   └── oss20b_inference.py   # OSS20B inference (--task flag)
│   └── ophthalmology/    # MCQ classification (5 scripts)
│       ├── hf_bench.py / gpt.py / openrouter.py
│       ├── cot_beams_score.py    # CoT beam search MCQ
│       └── oss20b_inference.py   # OSS20B local MCQ
├── csvs/                 # Results data
├── statistics/           # Statistical analysis scripts
│   ├── utils.py                  # Shared: Wilson CI, McNemar, bootstrap
│   ├── eurorad_stats.py          # Nominal: accuracy, McNemar, kappa
│   ├── nmed_stats.py             # Ordinal: Wilcoxon, ICC, weighted kappa
│   ├── export_csv.py             # Consolidated CSV export
│   └── all_statistics.csv        # All computed statistics
├── figures/              # Visualization scripts
│   ├── palette.py                # Shared color palette
│   ├── eurorad_accuracy_barplot.py
│   ├── eurorad_finetune_barplot.py
│   ├── eurorad_radar_plot.py
│   ├── eurorad_heatmap.py
│   ├── nmed_violin_plot.py
│   └── combined_figure.py        # Multi-panel fig2
├── finetune/             # LoRA fine-tuning
│   ├── omar_ft_hf.py             # gpt-oss-120b training (8-GPU MoE)
│   ├── qwen35b_finetune.py       # Qwen 3.5-35B training
│   └── train_wide.csv            # Training data
├── paper/                # LaTeX manuscript (synced to Overleaf)
├── data/                 # Dataset collection and preprocessing
└── gepa/                 # DSPy-based prompt optimization
```

## Key Technical Patterns

### Inference Providers
Models accessed via HF Inference Providers with provider suffix:
- `openai/gpt-oss-120b:fireworks-ai` / `:cerebras`
- `openai/gpt-oss-20b:<provider>`

### Beam Search Decoding (Eurorad)
```python
num_beams=13, num_beam_groups=13, diversity_penalty=0.5
num_return_sequences=13, max_new_tokens=3000, do_sample=False
```
**Majority voting**: Extract `<final>` label → normalize text → most frequent answer wins.

### Self-Consistency Framework
All benchmarks use k=3 independent runs per case:
- **Nominal tasks**: Majority vote (correct if ≥2/3 runs match)
- **Ordinal tasks**: Mean score consensus

### Resumable Benchmarking
All `hf_bench.py` scripts support `--resume` flag with incremental CSV appending.

## Statistical Analysis

Run all statistics: `PYTHONPATH=statistics python statistics/run_all.py`

Analyses include:
- Wilson Score 95% CIs for accuracy
- McNemar's test with Holm-Bonferroni correction (nominal)
- Wilcoxon signed-rank with Holm-Bonferroni correction (ordinal)
- Fleiss' kappa + bootstrap CIs (intra-model stability)
- Cohen's kappa + bootstrap CIs (inter-model agreement)
- ICC(3,k) + bootstrap CIs (ordinal stability)
- Linear weighted kappa (ordinal agreement)
- Per-subspecialty Fisher's exact test

## Figure Generation

All figures run from repo root: `python figures/<script>.py --output figures/<name>.png`

Combined paper figure: `python figures/combined_figure.py --output figures/combined.png`

## Text Normalization
- NFC Unicode normalization, lowercase, em/en-dash → hyphen
- Whitespace collapse, special character removal

## Environment
```bash
conda activate oss
pip install openai pandas tqdm transformers peft torch scipy scikit-learn matplotlib
```
