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
**On-device Gemma:** Gemma 4 31B, Gemma 4 26B

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
├── csvs/                 # Results data (see csvs/README.md)
│   ├── final_csvs/               # **Use this** — all fixes applied, Gemma included
│   ├── updated_csvs/             # Raw Gemma additions (Eurorad Qwen unfixed — provenance only)
│   └── finetune-eval/            # Per-checkpoint LoRA evaluation CSVs
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
└── gepa/                 # Prompt optimization
    ├── gepa_eurorad.py           # DSPy adapter (legacy, uses OpenAI)
    └── gepa_oa_eurorad.py        # optimize_anything for gpt-oss-120b (Bedrock)
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
- CP1252 mojibake recovery: `s.encode("cp1252").decode("utf-8")` (fixes Qwen outputs, no-op on clean text)
- NFKC Unicode normalization, lowercase, em/en-dash → hyphen
- Whitespace collapse, special character removal

## Data Quality Notes
See `claude/DATA_NOTES.md` for full details. Key issues (all fixed in `csvs/final_csvs/`):
- **Qwen row-shift**: 9 Eurorad columns had predictions rotated +1 row in cases 19041-19087. Affected: Qwen 35B, 35B FT, 9B FT. Not affected: Qwen 27B, 9B base, all other models.
- **Corrupted GT columns**: `FinalDiagnosisv2`, `v2_Alhusain`, `v3_Omar` removed. Use only `FinalDiagnosis`.
- **Qwen mojibake**: CP1252 encoding artifacts in all Qwen 3.5 outputs (~15-18 per run). Fixed via `norm_text()`.
- **Gemma 4**: Clean data, no issues found.

## GEPA Prompt Optimization

Automated prompt optimization using [GEPA](https://github.com/stanfordnlp/gepa) `optimize_anything`.
Complements fine-tuning: optimizes the system prompt without modifying model weights.

### Setup
```bash
pip install -e /path/to/gepa[full]    # GEPA with LiteLLM
# AWS credentials inherited from SageMaker execution role — no env vars needed
```

### Running
```bash
python gepa/gepa_oa_eurorad.py
```

### Architecture
- **Task model:** gpt-oss-120b via AWS Bedrock (`openai.gpt-oss-120b-1:0`)
- **Reflection LM:** Claude Opus 4.6 via AWS Bedrock (`anthropic.claude-opus-4-6-v1`) — analyzes failures and proposes prompt improvements
- **Dataset:** 1,895 train / 207 test Eurorad cases (from `finetune/` CSVs)
- **Metric:** Accuracy (exact match after normalization)
- **Budget:** 5,000 metric calls (up to 50 candidate proposals), minibatch size 5, disk-cached evaluations

Results saved to `gepa/runs/eurorad_oss120b/` (checkpoints, best prompt).

### Paper framing
Cloud reflection LM is used only during **one-time development** on published training data.
The optimized prompt deploys locally with the on-device model — no cloud dependency at inference.
This mirrors the fine-tuning pipeline where gpt-oss-120b (via Cerebras) generated training data
for gpt-oss-20b LoRA adaptation.

## Environment
```bash
conda activate oss
pip install openai pandas tqdm transformers peft torch scipy scikit-learn matplotlib
```
