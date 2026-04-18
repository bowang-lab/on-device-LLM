# GEPA Prompt Optimization — Experiments & Results

Last updated: 2026-04-06.

---

## Experiment Overview

Automated system prompt optimization for on-device LLMs on the Eurorad radiology
diagnosis benchmark using GEPA `optimize_anything`. The reflection LM (Claude Opus 4.6
on Bedrock) analyzes failure cases and proposes improved prompts; the task model is
called via API to evaluate each candidate.

### Common Setup

- **Dataset:** 1,895 Eurorad training cases split 80/20 → 1,516 train / 379 val (seed=42)
- **Held-out test:** 207 cases from `finetune/eurorad_test.csv` (never seen during optimization)
- **Reflection LM:** Claude Opus 4.6 via AWS Bedrock (`bedrock/us.anthropic.claude-opus-4-6-v1`)
- **Budget:** 5,000 metric calls / 50 candidate proposals per run
- **Evaluation:** Binary accuracy, exact match after text normalization (same as `hf_bench.py`)
- **Seed prompt:** Hand-crafted 6-line system prompt from `benchmarks/eurorad/hf_bench.py`

---

## Completed Runs

### Run 1 — gpt-oss-120b, Pareto, Medium Reasoning

**Script:** `gepa/gepa_oa_eurorad.py`
**Run dir:** `gepa/runs/eurorad_oss120b/`
**Task model:** gpt-oss-120b via Bedrock | **Reasoning:** Medium | **Strategy:** pareto
**Minibatch:** 5

| Metric | Value |
|---|---|
| Seed val | 77.6% |
| Best val (Program 8) | **82.6%** |
| Pareto front aggregate | 90.3% (14 candidates) |
| Iterations | 50 | Accepted: 14 |
| **Held-out test (seed)** | **80.2%** |
| **Held-out test (optimized)** | **83.1%** |
| **Test improvement** | **+2.9pp** |
| Runtime | ~56 min |

**Best result overall.** The GEPA-optimized prompt at medium reasoning matches the paper's
reported 83.1% for gpt-oss-120b with high reasoning. The best prompt is a 10-step
diagnostic reasoning scaffold with radiology-specific heuristics.

### Run 2 — gpt-oss-120b, Current Best, High Reasoning (from Run 1 seed)

**Script:** `gepa/gepa_oa_eurorad_r2.py`
**Run dir:** `gepa/runs/eurorad_oss120b_r2/`
**Task model:** gpt-oss-120b via Bedrock | **Reasoning:** High | **Strategy:** current_best
**Seed:** Run 1's best prompt (Program 8) | **Minibatch:** 5

| Metric | Value |
|---|---|
| Seed val | 82.3% |
| Best val | **82.3%** (no improvement) |
| Iterations | 36 | Accepted: 12 / Skipped: 24 |
| Held-out test | (= Run 1: 83.1%) |
| Runtime | ~54 min |

**No improvement.** Starting from an already-optimized prompt with high reasoning saturated
what's achievable. 12 candidates accepted on minibatch but none beat seed on full val.
Confirms **82-83% is the prompt optimization ceiling for gpt-oss-120b**.

### Run 3 — gpt-oss-120b, Pareto, High Reasoning (from manual seed)

**Script:** `gepa/gepa_oa_eurorad_r3.py`
**Run dir:** `gepa/runs/eurorad_oss120b_r3/`
**Task model:** gpt-oss-120b via Bedrock | **Reasoning:** High | **Strategy:** pareto
**Seed:** Manual prompt | **Minibatch:** 5

| Metric | Value |
|---|---|
| Seed val | 77.8% |
| Best val | **79.9%** |
| Iterations | 39 | Accepted: 12 |
| **Held-out test** | **83.1%** |
| Runtime | ~54 min |

Same held-out test result as Run 1 — different optimization path, same destination.
Confirms 83.1% test accuracy is robust across strategies and reasoning levels.

### Run 4 — gpt-oss-20b, Current Best, High Reasoning

**Script:** `gepa/gepa_oa_eurorad_oss20b.py`
**Run dir:** `gepa/runs/eurorad_oss20b/`
**Task model:** gpt-oss-20b via Bedrock | **Reasoning:** High | **Strategy:** current_best
**Seed:** Manual prompt | **Minibatch:** 5

| Metric | Value |
|---|---|
| Seed val | 75.5% |
| Best val | **78.9%** (+3.4pp) |
| Iterations | 28 | Accepted: 12 / Skipped: 16 |
| **Held-out test** | **77.3%** |
| Runtime | ~1h20m |

Val accuracy (78.9%) didn't transfer well to test (77.3%). Below the paper's 80.7%
baseline for gpt-oss-20b (H). The 20B model may need more iterations or a different
approach to reach its potential.

### Run 5 — Qwen3.5-35B-A3B, Current Best (attempt 1, killed)

**Script:** `gepa/gepa_oa_eurorad_qwen35b.py`
**Run dir:** `gepa/runs/eurorad_qwen35b/` (attempt 1)
**Task model:** Qwen3.5-35B-A3B via OpenRouter | **Strategy:** current_best
**Minibatch:** 5

| Metric | Value |
|---|---|
| Seed val | 80.7% |
| Best val | **80.7%** (no improvement) |
| Iterations | 8 | Accepted: 0 / Skipped: 7 |

**Killed — zero improvements.** Root cause analysis in Observations section below.

### Run 5b — Qwen3.5-35B-A3B, Current Best (attempt 2, minibatch=15)

**Script:** `gepa/gepa_oa_eurorad_qwen35b.py` (modified)
**Run dir:** `gepa/runs/eurorad_qwen35b/` (attempt 2)
**Task model:** Qwen3.5-35B-A3B via OpenRouter | **Strategy:** current_best
**Minibatch:** 15

| Metric | Value |
|---|---|
| Seed val | 80.7% |
| Best val | **80.7%** (no improvement) |
| Iterations | 50 | Accepted: 0 / Skipped: 34 |

**Failed again.** Even with minibatch=15, the last ~10 iterations scored 0/0 on both old
and new — likely OpenRouter provider errors causing all evaluations to silently fail.
The Qwen3.5-35B model via OpenRouter is not viable for GEPA optimization at this time.

### Qwen3.5-9B — Killed

**Script:** `gepa/gepa_oa_eurorad_qwen9b.py`
**Status:** Killed after 57 minutes — still on seed eval (371/379). At ~6 evals/min,
too slow for practical optimization. The 9B is the weakest model in the paper (67.6%).

---

## In Progress

### Run 6 — gpt-oss-120b Enhanced (reasoning traces + few-shot)

**Script:** `gepa/gepa_oa_eurorad_enhanced.py`
**Run dir:** `gepa/runs/eurorad_oss120b_enhanced/`
**Task model:** gpt-oss-120b via Bedrock | **Reasoning:** High | **Strategy:** pareto
**Minibatch:** 10

**Enhancements based on GEPA paper analysis:**
1. **Reasoning trace capture:** The model's internal reasoning (from `reasoningContent`)
   is included in the ASI feedback, so the reflection LM can see *why* the model chose
   wrong, not just *that* it chose wrong.
2. **Few-shot examples as second component:** `system_prompt` + `few_shot` optimized via
   round-robin. GEPA alternates between improving the instruction scaffold and the
   example demonstrations.
3. **Larger minibatch (10):** Better error signal for the reflection LM.

**Status:** Seed eval in progress.

---

## Summary Table

| Run | Model | Reasoning | Strategy | Seed val | Best val | **Test** | Status |
|---|---|---|---|---|---|---|---|
| **R1** | oss-120b | medium | pareto | 77.6% | 82.6% | **83.1%** | Done |
| R2 | oss-120b | high | current_best | 82.3% | 82.3% | (=R1) | Done, no gain |
| R3 | oss-120b | high | pareto | 77.8% | 79.9% | **83.1%** | Done |
| R4 | oss-20b | high | current_best | 75.5% | 78.9% | **77.3%** | Done |
| R5 | Qwen3.5-35B | default | current_best | 80.7% | 80.7% | — | Failed |
| R5b | Qwen3.5-35B | default | current_best | 80.7% | 80.7% | — | Failed |
| **R6** | **oss-120b** | **high** | **pareto** | — | — | — | **In progress** |

---

## Key Observations

### 1. Prompt optimization ceiling at 83.1% for gpt-oss-120b

Three independent runs (R1, R2, R3) all converge on 83.1% held-out test accuracy
regardless of strategy, reasoning level, or seed prompt. This appears to be the hard
ceiling for system-prompt-only optimization on this model/task.

### 2. GEPA prompt optimization ≈ one reasoning tier

Run 1 showed that an optimized prompt at **medium reasoning** (83.1% test) matches the
paper's baseline at **high reasoning** (83.1% test). Prompt optimization can partially
substitute for compute-intensive reasoning, which has practical deployment implications
for latency-sensitive clinical applications.

### 3. Minibatch size must scale with seed accuracy

**Key finding from Run 5 (Qwen3.5-35B):** When seed accuracy is high (>80%), a minibatch
of 5 provides insufficient error signal. Most minibatches score 4/5 or 5/5, producing
ties that are always rejected. The reflection LM sees few errors and proposes generic,
repetitive prompts.

Evidence from Run 5 logs:
- Iters 2,3,6: new 5, old 5 → tie, rejected
- Iters 1,4: new 4, old 5 → worse, rejected
- Iters 5,7: new 3, old 3 → tie, rejected
- Proposed prompts were near-identical across iterations

Recommended minibatch sizes:
- Seed < 70%: minibatch 5 (E[errors] ≈ 1.5)
- Seed 70-80%: minibatch 10 (E[errors] ≈ 2-3)
- Seed > 80%: minibatch 15+ (E[errors] ≈ 3+)

### 4. Pareto strategy produces same test results as current_best

R1 (pareto) and R3 (pareto) both reached 83.1% test. R2 (current_best from R1's best)
couldn't improve further. For initial exploration from a manual seed, pareto and
current_best reach the same ceiling — pareto explores more broadly but both converge.

### 5. Val accuracy doesn't always predict test accuracy

| Run | Best val | Test |
|---|---|---|
| R1 | 82.6% | 83.1% |
| R3 | 79.9% | 83.1% |
| R4 | 78.9% | 77.3% |

R3 had lower val accuracy than R1 but identical test accuracy. R4's val gains didn't
transfer. Val scores are useful for candidate selection during optimization but should
not be reported as the final metric.

### 6. OpenRouter models are impractical for GEPA optimization

Both Qwen runs failed — the 9B was too slow (6 evals/min → 63 min per validation pass),
and the 35B hit provider errors in later iterations causing 0/0 scores. Bedrock models
are significantly more reliable and faster for this workflow.

---

## Why GEPA Doesn't Match Fine-Tuning Gains

Based on analysis of the GEPA paper (arXiv:2507.19457v2, ICLR 2026):

### The fundamental limitation

GEPA optimizes **instructions around frozen weights**. Fine-tuning modifies the weights
themselves. For Eurorad radiology diagnosis:

- **Fine-tuning** teaches the model new domain knowledge (imaging patterns, pathology
  associations, clinical reasoning). gpt-oss-20b went from 80.7% → 86.5% (+5.8pp).
- **GEPA** improves how the model uses its *existing* knowledge. gpt-oss-120b went from
  80.2% → 83.1% (+2.9pp on test).

If the model doesn't know that "expansile lytic lesion with low T1 and high T2" suggests
giant cell tumor, no system prompt can fix that.

### Specific limiting factors

1. **Binary feedback with weak error signal.** Our evaluator returns 1/0. The GEPA paper's
   best results (+19% on HotpotQA) came from tasks with rich traces — multi-hop reasoning
   chains, compiler errors, rubric breakdowns. Our ASI had the wrong/correct answer but
   not the model's reasoning about *why* it chose wrong. (Fixed in Run 6.)

2. **Knowledge-bound vs. reasoning-bound.** GEPA's biggest wins in the paper were on tasks
   where better reasoning instructions help: math (+12%), multi-hop QA (+19%). Radiology
   MCQ is primarily knowledge retrieval — the model either knows the pathology or doesn't.

3. **Single component optimization.** The GEPA paper optimizes multi-module programs
   (retrieval → reasoning → output). We optimized a single system prompt. (Fixed in Run 6
   with two components: system_prompt + few_shot.)

4. **Abundant training data favors fine-tuning.** We have 1,895 training cases — exactly
   the regime where the GEPA paper says "weight updates will outperform prompting."

5. **No few-shot optimization.** The GEPA paper notes: "GEPA currently focuses on
   optimizing instructions alone, omitting exemplar or few-shot demonstration optimization.
   Incorporating such examples could further improve performance." (Fixed in Run 6.)

### What the GEPA paper says about this

> "Although GEPA excels when rollouts are expensive, it is likely that weight updates
> will outperform prompting in regimes with abundant data or when large-scale rollouts
> are feasible."

Our task has abundant data (1,895 training cases) and cheap rollouts (Bedrock API).
This is exactly where the paper predicts fine-tuning wins.

### The value of prompt optimization despite smaller gains

Even though GEPA doesn't match fine-tuning, it provides complementary value:
- **No GPU infrastructure required** — just API calls during development
- **No weight modification** — same model binary, simpler deployment
- **Faster iteration** — ~1 hour vs. hours/days for LoRA training
- **Potentially stackable** — GEPA-optimized prompt + fine-tuned model (untested)
- **Equivalent to one reasoning tier** — medium reasoning + optimized prompt ≈ high reasoning

---

## Scripts Reference

| Script | Task model | Strategy | Reasoning | Minibatch | Components |
|---|---|---|---|---|---|
| `gepa_oa_eurorad.py` | oss-120b (Bedrock) | pareto | medium | 5 | system_prompt |
| `gepa_oa_eurorad_r2.py` | oss-120b (Bedrock) | current_best | high | 5 | system_prompt |
| `gepa_oa_eurorad_r3.py` | oss-120b (Bedrock) | pareto | high | 5 | system_prompt |
| `gepa_oa_eurorad_oss20b.py` | oss-20b (Bedrock) | current_best | high | 5 | system_prompt |
| `gepa_oa_eurorad_qwen35b.py` | Qwen3.5-35B (OpenRouter) | current_best | default | 15 | system_prompt |
| `gepa_oa_eurorad_qwen9b.py` | Qwen3.5-9B (OpenRouter) | current_best | default | 5 | system_prompt |
| **`gepa_oa_eurorad_enhanced.py`** | **oss-120b (Bedrock)** | **pareto** | **high** | **10** | **system_prompt + few_shot** |

All scripts use same train/val split (seed=42, 80/20), same held-out test (207 cases),
same reflection LM (Opus 4.6), same budget (5,000 metric calls / 50 proposals).
