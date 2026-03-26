# On-Device Large Language Models
Benchmarking OSS model performance on-device for medical tasks.

## 0. Setup & Data
Please download the data from the following link and place the csv files under `data/datasets`: https://drive.google.com/drive/folders/19C-Eey_yYKk1sIJc1MxFolIWwH_TBHfI?usp=sharing 

---

## 1. NMED Benchmarks (Diagnosis & Treatment)

### GPT-OSS (HuggingFace API)
Run the `hf_bench.py` script for `gpt-oss-20b` or `120b` across different reasoning efforts (`low`, `medium`, `high`).

**Template Command:**
```bash
python benchmarks/nmed-notes/hf_bench.py \
  data/datasets/nmed_diagnosis.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --api chat \
  --reasoning_effort high \
  --max_output_tokens 8192 \
  --workers 1 \
  --results results \
  --resume \
  --output_csv nmed-diagnosis-oss-120b-high-v1.csv
```

**Configurations:**
*   **Dataset:** Swap `nmed_diagnosis.csv` with `nmed_treatment.csv`.
*   **Model:** `openai/gpt-oss-120b:fireworks-ai` or `openai/gpt-oss-20b:fireworks-ai`.
*   **Effort:** `--reasoning_effort` can be `low`, `medium`, or `high`.
*   **Task Mode:** For treatment datasets, add flag `--eval_mode treatment`.

### Proprietary (OpenAI API)
```bash
# GPT-5 / GPT-4o / o4-mini
python3 benchmarks/nmed-notes/gpt.py \
  --model gpt-5-2025-08-07 \
  --dataset data/datasets/nmed_diagnosis.csv \
  --results-dir results \
  --resume \
  --output-csv nmed-diagnosis-gpt-5-2025-08-07-v2.csv \
  --eval-mode diagnosis
```

### OpenRouter (DeepSeek)
```bash
python benchmarks/nmed-notes/openrouter.py \
  data/datasets/nmed_treatment.csv \
  --endpoint deepseek/deepseek-r1-0528 \
  --results_dir results \
  --max_output_tokens 8192 \
  --workers 1 \
  --resume \
  --output_csv nmed_deepseek/treatment-deepseek-r1-0528_v1.csv
```

---

## 2. Eurorad Benchmarks

### Proprietary (OpenAI API)
```bash
# Standard Batch Mode
python benchmarks/eurorad/gpt.py --mode batch --debug

# Chat Mode (Specific Snapshot)
python3 benchmarks/eurorad/gpt.py \
  --model gpt-5-2025-08-07 \
  --dataset data/datasets/eurorad_test.csv \
  --results-dir results \
  --resume \
  --output-csv eurorad_gpt-5-2025-08-07_v4.csv
```

### GPT-OSS (HuggingFace API)
```bash
python benchmarks/eurorad/hf_bench.py \
  data/datasets/eurorad_test.csv \
  --model openai/gpt-oss-20b:fireworks-ai \
  --api chat \
  --reasoning_effort low \
  --max_output_tokens 8192 \
  --workers 1 \
  --results results \
  --resume \
  --output_csv oss20b_low_v3.csv
```

### OpenRouter (DeepSeek)
```bash
python benchmarks/eurorad/openrouter.py \
  data/datasets/eurorad_test.csv \
  --endpoint deepseek/deepseek-r1-0528 \
  --results_dir results \
  --max_output_tokens 8192 \
  --workers 1 \
  --resume \
  --output_csv eurorad_deepseek/deepseek-r1-0528_v3.csv
```

### Fine-Tuned GPT-OSS-20B (LoRA)
```bash
python benchmarks/eurorad/oss20b_inference.py \
  --base-model openai/gpt-oss-20b \
  --lora-path /path/to/lora_adapter \
  --test-csv data/datasets/eurorad_val.csv \
  --model-name gptoss20b_finetuned \
  --num-beam-groups 13 \
  --diversity-penalty 0.5 \
  --max-new-tokens 3000 \
  --cuda-visible-devices 0
```
*   Uses diverse beam search with majority voting for diagnosis selection.
*   Requires pre-trained LoRA adapter from fine-tuning step.

---

## 3. Ophthalmology Benchmarks

### GPT-OSS (20b & 120b)
**Template Command:**
```bash
python benchmarks/ophthalmology/hf_bench.py \
  data/datasets/ophthalmology.csv \
  --model openai/gpt-oss-20b:fireworks-ai \
  --api chat \
  --reasoning_effort low \
  --max_output_tokens 8192 \
  --workers 2 \
  --results results \
  --resume \
  --output_csv results/ophthalmology_oss20b_low_v1.csv
```
*   **Variations:** Change `--model` to `120b`, `--reasoning_effort` to `medium`/`high`, and increment output filenames (`v1`, `v2`, `v3`).

### Proprietary (OpenAI)
```bash
# Responses Mode (GPT-5 / o4-mini)
python3 benchmarks/ophthalmology/gpt.py \
  --model gpt-5-2025-08-07 \
  --dataset data/datasets/ophthalmology.csv \
  --results-dir results \
  --resume \
  --output-csv ophthalmology_responses_gpt-5-2025-08-07_v1.csv
```

### Other Providers (Novita / OpenRouter)
```bash
# Novita (Baichuan)
python benchmarks/ophthalmology/novita.py data/datasets/ophthalmology.csv \
  --endpoint baichuan/baichuan-m2-32b \
  --sleep 1.2 --timeout 60 --verbose --resume

# OpenRouter (Qwen / Llama)
python benchmarks/ophthalmology/openrouter.py data/datasets/ophthalmology.csv \
  --endpoint qwen/qwen3-235b-a22b-2507 \
  --results_dir results \
  --workers 1 \
  --resume \
  --output_csv results/qwen/qwen3-235b-a22b-2507-v2.csv
```

---

## 4. Data Processing Utilities

**Eurorad Data Selection**
```bash
# Convert raw CSVs to JSON
python data/csvs_to_json.py --indir eurorad_csvs --out eurorad_cases.json

# Combine into wide format
python data/combine_cases_csv.py --indir eurorad_csvs --out eurorad_cases_wide.csv
```

**Eurorad Scrapers**
```bash
# Get single case
python data/get_case_eurorad.py https://www.eurorad.org/case/18706

# Get 2025 range
python data/get_range_eurorad.py --start 18806 --end 19164 --outdir eurorad_csvs

# Get training cases
python data/get_range_eurorad.py --csv data/eurorad_train_cases.csv --case-id-col "Case ID" --outdir eurorad_train_csvs --resume
```

---

## 5. Diverse Beam Search Evaluation (Local Inference)

Self-consistency via diverse beam search with majority voting. Used for gpt-oss-120b and fine-tuned models.

### Eurorad
```bash
# gpt-oss-120b with 13-beam diverse beam search
CUDA_VISIBLE_DEVICES=0 python benchmarks/eurorad/eurorad_beams_hf.py \
  --input_csv data/datasets/eurorad_test.csv \
  --output_csv results/eurorad_oss120b_13beams_v1.csv \
  --model openai/gpt-oss-120b \
  --num_beams 13 --num_beam_groups 13

# Chain-of-thought beams with external rescoring
python benchmarks/eurorad/cot_beams_score_hf.py \
  --input_csv data/datasets/eurorad_test.csv \
  --output_csv results/eurorad_cot_beams.csv \
  --model openai/gpt-oss-120b

# Evaluate fine-tuned LoRA checkpoint
python benchmarks/eurorad/eval_finetune.py \
  --base_model openai/gpt-oss-120b \
  --lora_path finetune/outputs/checkpoint-30 \
  --test_csv data/datasets/eurorad_test.csv
```

### NMED (Diagnosis & Treatment)
```bash
# Diagnosis scoring with beam search
python benchmarks/nmed-notes/beams_diagnosis_hf.py \
  --val-csv data/datasets/nmed_diagnosis.csv \
  --out-csv results/nmed_diagnosis_oss120b_5beams_v1.csv \
  --num-beams 5 --max-new-tokens 3000

# Treatment scoring with beam search
python benchmarks/nmed-notes/beams_treatment_hf.py \
  --val-csv data/datasets/nmed_treatment.csv \
  --out-csv results/nmed_treatment_oss120b_5beams_v1.csv \
  --num-beams 5 --max-new-tokens 3000
```

---

## 6. Fine-Tuning (LoRA)

### Primary: gpt-oss-120b SFT with LoRA on 8 GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune/omar_ft_hf.py \
  MODEL_ID=openai/gpt-oss-120b \
  MAX_SEQ_LEN=2816 \
  LORA_R=4 LORA_ALPHA=8 LORA_DROPOUT=0.0 \
  EXPERT_LAYERS=33 EXPERT_TOKENS=gate_up_proj \
  BATCH_PER_DEVICE=1 GRAD_ACCUM=16 NUM_EPOCHS=3 \
  LR=6e-5 WD=0.01 WARMUP_RATIO=0.1
```

Training data is in `finetune/` (JSONL and CSV files). See notebooks for exploratory fine-tuning:
- `finetune/oss-eurorad-sft.ipynb` — Eurorad SFT
- `finetune/V7Finetuning_Gptoss120Reasonthinking_Final_Omar.ipynb` — MoE expert fine-tuning
- `finetune/Finetuning_MoELinear_oss120ReasData_detPrompt.ipynb` — MoE Linear with reasoning data

---

## 7. GEPA Prompt Optimization

DSPy-based prompt optimization for Eurorad diagnosis:
```bash
python gepa/gepa_eurorad.py
```

---

## 8. Figure Generation

Scripts for paper figures in `figures/`:
- `figures/plot_radar.py` — Radar chart (multi-metric model comparison)
- `figures/kendall_tau_violin_plot.py` — Kendall's tau violin plot (input: `figures/fig2-c-box-plot/kendall.xlsx`)
- `figures/visualization_barplot.py` — Grouped bar plot for hyperparameter tuning