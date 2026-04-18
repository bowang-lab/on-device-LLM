# GEPA Prompt Optimization for Eurorad

Automated system prompt optimization for gpt-oss-120b on the Eurorad radiology
diagnosis benchmark using GEPA `optimize_anything`.

## Motivation

The paper establishes three adaptation strategies for on-device models:

| Strategy | Modifies weights | Cloud at inference | Development cost |
|----------|------------------|--------------------|------------------|
| Manual prompt engineering | No | No | Free |
| **GEPA prompt optimization** | **No** | **No** | **Low (API calls)** |
| LoRA fine-tuning | Yes | No | High (GPU hours) |

Prompt optimization sits between manual prompting and fine-tuning: it systematically
searches for better prompts using LLM reflection on failure cases, without modifying
model weights. The optimized prompt deploys identically to a hand-written one.

## How it works

GEPA `optimize_anything` runs an iterative loop:

1. **Evaluate** the current system prompt on a minibatch of training cases
2. **Capture traces** — for each case, record the model's output, the correct answer,
   and whether it matched
3. **Reflect** — a reflection LM (Claude Sonnet 4 on Bedrock) analyzes the failures:
   which diagnoses were confused, what patterns led to errors, how the prompt could
   better guide reasoning
4. **Propose** — the reflection LM suggests an improved system prompt
5. **Accept/reject** — if the new prompt scores higher on the minibatch, promote it;
   then validate on the full test set
6. **Maintain Pareto front** — track multiple non-dominated candidates

## Script: `gepa/gepa_oa_eurorad.py`

### Components

- **Task model:** gpt-oss-120b via AWS Bedrock (`bedrock/us.openai.gpt-oss-120b-1:0`)
  - Max output tokens: 512
  - Called via LiteLLM → Bedrock

- **Reflection LM:** Claude Opus 4.6 via AWS Bedrock (`bedrock/us.anthropic.claude-opus-4-6-v1`)
  - Analyzes failure patterns and proposes improved system prompts
  - Up to 50 reflection calls over a 5,000 metric-call budget

- **Seed prompt:** The hand-crafted system prompt from `benchmarks/eurorad/hf_bench.py`

- **User prompt template:** Fixed (not optimized). Contains case description + options list.

- **Evaluator:** Binary accuracy via exact match after text normalization. Returns
  actionable side information (model output, expected answer, match type, feedback).

- **Dataset:**
  - Train: 1,895 pre-2025 Eurorad cases (`finetune/eurorad_train.csv`)
  - Test: 207 held-out 2025 cases (`finetune/eurorad_test.csv`)

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_metric_calls` | 5000 | ~23 accepted candidates with full 207-example validation each |
| `max_candidate_proposals` | 50 | Hard cap on total proposals (accepted + rejected) |
| `reflection_minibatch_size` | 5 | Enough cases per reflection for the LM to spot patterns |
| `cache_evaluation` | True (disk) | Avoids redundant gpt-oss-120b calls on same (prompt, case) pairs |
| `max_workers` | 8 | Parallel evaluation across training cases |
| `candidate_selection_strategy` | pareto (default) | Maintains diversity across candidates |
| `merge` | 3 invocations | Cross-pollination between Pareto-optimal prompts |
| `module_selector` | all | Only one component (system_prompt), so all == round_robin |

### Running

```bash
# Install GEPA
pip install -e /path/to/gepa[full]

# Run optimization (AWS credentials inherited from SageMaker execution role)
python gepa/gepa_oa_eurorad.py
```

### Output

Results are saved to `gepa/runs/eurorad_oss120b/`:
- `best_system_prompt.txt` — the best prompt found
- GEPA checkpoints (for resuming interrupted runs)
- Evaluation cache (disk-backed)

## Paper integration

### Framing

The use of a cloud reflection LM (Claude Sonnet 4) during development is consistent
with the paper's existing methodology:

- Fine-tuning gpt-oss-20b used **gpt-oss-120b via Cerebras** (cloud) to generate
  training data
- Fine-tuning Qwen3.5-35B used **Qwen3.5-122B** (cloud) to curate training data

In both cases, cloud resources were used during one-time development on published
(non-patient) data, producing artifacts that deploy locally. GEPA prompt optimization
follows the same pattern: Claude Opus 4.6 on Bedrock reflects on failures during
development, producing an optimized prompt that deploys locally with no cloud dependency.

### Expected results

- Baseline gpt-oss-120b (H): 83.1% accuracy on Eurorad test set
- Target: 85-87% with optimized prompt (closing gap toward GPT-5.1 at 88.9%)
- The result demonstrates that lightweight adaptation via prompt optimization can
  improve on-device models without fine-tuning infrastructure

### Comparison table for paper

| Adaptation | gpt-oss-20b | gpt-oss-120b |
|------------|-------------|--------------|
| Zero-shot (manual prompt) | 80.7% | 83.1% |
| GEPA prompt optimization | — | TBD |
| LoRA fine-tuning | 86.5% | — |

## Differences from `gepa_eurorad.py` (legacy)

The original `gepa_eurorad.py` uses:
- DSPy adapter with full program evolution (modifies DSPy module source code)
- OpenAI models (gpt-4.1-mini task, gpt-4.1 reflection)
- HuggingFace datasets hub for data loading

The new `gepa_oa_eurorad.py` uses:
- `optimize_anything` API (simpler, no DSPy dependency)
- Bedrock Claude Opus 4.6 for reflection (AWS-native)
- gpt-oss-120b as the actual task model being optimized
- Local CSV files for data loading
- Only the system prompt text is optimized (not code or program structure)
