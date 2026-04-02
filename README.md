# On-Device Large Language Models for Clinical Decision Support

Benchmarking and fine-tuning on-device LLMs (gpt-oss and Qwen3.5 families) across three medical benchmarks: general radiology diagnosis, ophthalmology specialty QA, and clinical judgment simulation.

## Setup

```bash
conda activate oss
pip install openai pandas tqdm transformers peft torch scipy scikit-learn matplotlib
```

Download datasets and place under `data/datasets/`:
https://drive.google.com/drive/folders/19C-Eey_yYKk1sIJc1MxFolIWwH_TBHfI?usp=sharing

## Directory Structure

```
on-device-LLM/
├── benchmarks/              # Evaluation scripts
│   ├── eurorad/             # Radiology diagnosis (8 scripts)
│   ├── nmed-notes/          # Diagnosis & treatment Likert scoring (6 scripts)
│   └── ophthalmology/       # Eye disease MCQ classification (5 scripts)
├── csvs/                    # Results (raw prediction data)
│   ├── Eurorad.csv          # 207 test cases, 10 models x 3 runs
│   ├── Ophthalmology.csv    # 130 questions
│   ├── NMED_Diagnosis.csv   # 720 cases
│   └── NMED_Treatment.csv   # 595 cases
├── statistics/              # Statistical analysis pipeline
│   ├── eurorad_stats.py     # Accuracy, McNemar, Fleiss/Cohen kappa
│   ├── nmed_stats.py        # Wilcoxon, ICC, weighted kappa
│   ├── export_csv.py        # Consolidated CSV export
│   └── all_statistics.csv   # All computed statistics (400 rows)
├── figures/                 # Visualization scripts
│   ├── palette.py           # Shared color palette
│   ├── combined_figure.py   # Multi-panel paper figure
│   └── (5 individual plot scripts)
├── finetune/                # LoRA fine-tuning
│   ├── omar_ft_hf.py        # gpt-oss-120b (8-GPU MoE)
│   ├── qwen35b_finetune.py  # Qwen 3.5-35B
│   └── *.csv                # Training data
├── paper/                   # LaTeX manuscript (Overleaf sync)
├── data/                    # Dataset collection
└── gepa/                    # DSPy prompt optimization
```

## Model Lineup

| Category | Models |
|----------|--------|
| **Proprietary** | GPT-5.1, GPT-5-mini, Gemini 3.1 Pro |
| **Open-source** | DeepSeek-R1 |
| **On-device gpt-oss** | gpt-oss-20b (H), gpt-oss-120b (H) |
| **On-device Qwen3.5** | 9B, 27B, 35B, 35B (fine-tuned) |

## Running Benchmarks

All benchmark scripts support `--resume` to skip already-completed rows.

### HuggingFace Inference Providers (gpt-oss, Qwen)

```bash
# Eurorad
python benchmarks/eurorad/hf_bench.py data/datasets/eurorad_test.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --api chat --reasoning_effort high --max_output_tokens 8192 --resume

# NMED (swap diagnosis/treatment CSV; add --eval_mode treatment for treatment)
python benchmarks/nmed-notes/hf_bench.py data/datasets/nmed_diagnosis.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --eval_mode diagnosis --reasoning_effort high --max_output_tokens 8192 --resume

# Ophthalmology
python benchmarks/ophthalmology/hf_bench.py data/datasets/ophthalmology.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --api chat --reasoning_effort high --max_output_tokens 8192 --resume
```

### OpenAI API (GPT-5.1)

```bash
python benchmarks/eurorad/gpt.py \
  --model gpt-5.1-2025-11-13 --dataset data/datasets/eurorad_test.csv \
  --results-dir csvs --resume
```

### OpenRouter (DeepSeek, Qwen, etc.)

```bash
python benchmarks/eurorad/openrouter.py data/datasets/eurorad_test.csv \
  --endpoint deepseek/deepseek-r1-0528 --results_dir csvs --resume
```

### Local Inference (Diverse Beam Search)

```bash
# Eurorad (13 beams)
CUDA_VISIBLE_DEVICES=0 python benchmarks/eurorad/eurorad_beams_hf.py \
  --input_csv data/datasets/eurorad_test.csv \
  --output_csv csvs/eurorad_oss120b_13beams_v1.csv \
  --model openai/gpt-oss-120b \
  --num_beams 13 --num_beam_groups 13

# NMED beam search (diagnosis or treatment)
python benchmarks/nmed-notes/beams_hf.py \
  --task diagnosis \
  --val-csv data/datasets/nmed_diagnosis.csv \
  --out-csv csvs/nmed_diagnosis_beams_v1.csv
```

## Fine-Tuning

### gpt-oss-120b (LoRA, 8 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune/omar_ft_hf.py \
  MODEL_ID=openai/gpt-oss-120b \
  MAX_SEQ_LEN=2816 \
  LORA_R=4 LORA_ALPHA=8 LORA_DROPOUT=0.0 \
  EXPERT_LAYERS=33 EXPERT_TOKENS=gate_up_proj \
  BATCH_PER_DEVICE=1 GRAD_ACCUM=16 NUM_EPOCHS=3 \
  LR=6e-5 WD=0.01 WARMUP_RATIO=0.1
```

### Qwen 3.5-35B (LoRA)

```bash
python finetune/qwen35b_finetune.py
```

## Statistical Analysis

```bash
# Run all analyses (Eurorad + NMED)
PYTHONPATH=statistics python statistics/run_all.py

# Export consolidated CSV
PYTHONPATH=statistics python statistics/export_csv.py
```

Computes: Wilson CIs, McNemar + Holm-Bonferroni, Wilcoxon + Holm-Bonferroni, Fleiss/Cohen kappa with bootstrap CIs, ICC(3,k), weighted kappa, Fisher's exact per subspecialty.

## Figure Generation

```bash
# Individual plots
python figures/eurorad_accuracy_barplot.py -o figures/eurorad_accuracy.png
python figures/eurorad_finetune_barplot.py -o figures/eurorad_finetune.png
python figures/eurorad_radar_plot.py -o figures/eurorad_radar.png
python figures/eurorad_heatmap.py -o figures/eurorad_heatmap.png
python figures/nmed_violin_plot.py -o figures/nmed_violin.png

# Combined paper figure (fig2)
python figures/combined_figure.py -o figures/combined.png
```
