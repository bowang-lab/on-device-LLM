# On-Device Large Language Models
Benchmarking OSS model performance on-device for medical tasks.

## 1. NMED Benchmarks (Diagnosis & Treatment)

### GPT-OSS (HuggingFace API)
Run the `hf_bench.py` script for `gpt-oss-20b` or `120b` across different reasoning efforts (`low`, `medium`, `high`).

**Template Command:**
```bash
python benchmarks/nmed-notes-score/hf_bench.py \
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
python3 benchmarks/nmed-notes-score/gpt.py \
  --model gpt-5-2025-08-07 \
  --dataset data/datasets/nmed_diagnosis.csv \
  --results-dir results \
  --resume \
  --output-csv nmed-diagnosis-gpt-5-2025-08-07-v2.csv \
  --eval-mode diagnosis
```

### OpenRouter (DeepSeek)
```bash
python benchmarks/nmed-notes-score/openrouter.py \
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

## 5. Finetuning

```bash
conda activate oss
CUDA_VISIBLE_DEVICES=0,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune/oss20b.py
```