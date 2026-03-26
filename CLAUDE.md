# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**oss-benchmark**: Evaluates large language models on medical diagnostic tasks, primarily using OpenAI's gpt-oss-120b and gpt-oss-20b models.

### Primary Use Case
The repo is actively used for experimentation with the `openai/gpt-oss-120b` model (and 20B variant) on three medical benchmarks:
1. **EuroRAD** - Radiology case diagnosis (multiple-choice selection)
2. **NMED** - Diagnosis and treatment scoring using Likert rubrics (1-5 scale)
3. **Ophthalmology** - Eye disease diagnostic classification

## High-Level Architecture

```
os-benchmark/
├── benchmarks/           # Evaluation scripts for each domain
│   ├── eurorad/        # Radiology case evaluation
│   │   ├── hf_bench.py          # HF Inference Provider API benchmarking
│   │   ├── eurorad_beams_hf.py  # Local model: diverse beam search + majority voting
│   │   ├── cot_beams_score_hf.py # CoT beams with external log-prob rescoring
│   │   └── eval_finetune.py      # Evaluation for fine-tuned models
│   ├── nmed-notes-score/
│   │   ├── hf_bench.py         # Likert scoring (1-5) via HF Inference Providers
│   │   └── gpt.py              # OpenAI API benchmarking (gpt-5, o4-mini)
│   └── ophthalmology/        # Eye disease classification
├── data/               # Dataset collection and preprocessing
├── finetune/           # LoRA fine-tuning scripts
│   ├── omar_ft_hf.py              # 8-GPU MoE training script
│   ├── omar20b_ft_hf.py           # 20B variant version
│   └── train_wide.csv             # Training data reference
├── gepa/               # GPT-OSS Evaluation & Alignment framework
└── figures/            # Visualization scripts for results
```

## Key Technical Patterns

### Inference Providers Model
Models are accessed via Hugging Face Inference Providers with provider suffix:
- `openai/gpt-oss-120b:fireworks-ai`
- `openai/gpt-oss-120b:cerebras`
- `openai/gpt-oss-20b:<provider>`

### Beam Search Decoding (EuroRAD)
Default configuration for local model evaluation:
```python
num_beams=13           # or 9
num_beam_groups=13     # 1 group per beam = max diversity
diversity_penalty=0.5
num_return_sequences=13
max_new_tokens=3000
do_sample=False        # deterministic generation
```
**Majority voting**: Extract `<final>` label from each beam → normalize text → most frequent answer wins (tie → earliest beam).

### Harmony Channel Prompt Format
Models expected to output in structured format:
```xml
<|channel|>analysis<|message|>
[Detailed clinical reasoning (~150-300 words)]
<|return|>
<|channel|>final<|message|>
[Glioblastoma]   # verbatim label from list, no punctuation
<|return|>
```
The system extracts `<analysis>` for reasoning traces and `<final>` for the answer.

### Resumable Benchmarking
All `hf_bench.py` scripts support `--resume` flag. They:
- Track processed case_ids in output CSV
- Skip already-completed rows on re-run
- Incrementally append results per case

## Running Benchmarks

### EuroRAD (Local Model with Beam Search)
```bash
conda activate oss
cd benchmarks/eurorad
CUDA_VISIBLE_DEVICES=0 python eurorad_beams_hf.py \
  --input_csv data/datasets/eurorad_test.csv \
  --output_csv results/eurorad_oss120b_13beams_v1.csv \
  --model openai/gpt-oss-120b \
  --num_beams 13 --num_beam_groups 13
```

### EuroRAD (HF Inference Provider API)
```bash
python hf_bench.py eurorad_test.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --api chat --reasoning_effort high \
  --max_output_tokens 8192 --workers 4
```

### NMED Diagnosis Scoring
```bash
python nmed-notes-score/hf_bench.py \
  data/datasets/nmed_diagnosis.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --eval_mode diagnosis --reasoning_effort high \
  --max_output_tokens 8192
```

### NMED Treatment Scoring
```bash
python nmed-notes-score/hf_bench.py \
  data/datasets/nmed_treatment.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --eval_mode treatment --reasoning_effort high \
  --max_output_tokens 8192
```

## Finetuning Scripts

### GPT-OSS-120B with LoRA (8 GPUs)
```bash
cd finetune
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python omar_ft_hf.py \
  MODEL_ID=openai/gpt-oss-120b \
  MAX_SEQ_LEN=2816 \
  LORA_R=4 LORA_ALPHA=8 LORA_DROPOUT=0.0 \
  EXPERT_LAYERS=33 EXPERT_TOKENS=gate_up_proj \
  BATCH_PER_DEVICE=1 GRAD_ACCUM=16 NUM_EPOCHS=3 \
  LR=6e-5 WD=0.01 WARMUP_RATIO=0.1
```

## Data Files

Expected CSV format for EuroRAD:
```csv
case_id,PostDescription,DifferentialDiagnosisList,FinalDiagnosis
12345,"Patient description...","Option A, Option B, Option C","Option B"
```

## Output Metrics
- **Accuracy**: Normalized string match between model prediction and ground truth
- **Match Type**: `exact`, `normalized`, `fuzzy` (80%+ similarity), or `no_match`
- **Beam Agreement**: Percentage of beams producing identical answer before selection

## Text Normalization
- NFC Unicode normalization
- Lowercase conversion
- Em-dash/en-dash → hyphen conversion
- Whitespace collapse
- Special character removal for comparison

## Environment Requirements
```bash
conda activate oss
pip install openai pandas tqdm transformers peft torch
```

## File Naming Convention for Results
Results follow pattern: `{model}_{numbeams}beams_v{version}.csv` (e.g., `eurorad_oss120b_13beams_v4.csv`).
