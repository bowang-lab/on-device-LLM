#!/usr/bin/env python3
"""
GEPA optimize_anything on Eurorad — Enhanced run with:
  1. Reasoning trace capture: model's internal reasoning included in ASI feedback
  2. Few-shot examples as a second optimizable component

Task model:      gpt-oss-120b via Bedrock (Reasoning: high)
Reflection LM:   Claude Opus 4.6 via Bedrock
Strategy:        pareto (better for multi-component optimization)
"""

import random
import re
import time
import difflib
import unicodedata
from pathlib import Path
from typing import Any

import litellm
import pandas as pd

import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    MergeConfig,
    optimize_anything,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TASK_MODEL = "bedrock/openai.gpt-oss-120b-1:0"
REFLECTION_LM = "bedrock/us.anthropic.claude-opus-4-6-v1"

MAX_OUTPUT_TOKENS = 2048

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "finetune" / "eurorad_train.csv"
TEST_CSV = PROJECT_ROOT / "finetune" / "eurorad_test.csv"
RUN_DIR = Path(__file__).resolve().parent / "runs" / "eurorad_oss120b_enhanced"

VAL_FRACTION = 0.2
SPLIT_SEED = 42

MAX_METRIC_CALLS = 5000
MAX_CANDIDATE_PROPOSALS = 50
REFLECTION_MINIBATCH_SIZE = 10  # larger minibatch for better error signal at high accuracy
MAX_RETRIES = 3
BASE_BACKOFF = 2.0

# ---------------------------------------------------------------------------
# Seed components
# ---------------------------------------------------------------------------

SEED_SYSTEM_PROMPT = (
    "You are a careful radiology diagnosis selector.\n"
    "Given a case description and a finite list of candidate diagnoses, "
    "select the single most likely final diagnosis FROM THE LIST.\n"
    "Response rules:\n"
    "1) Output EXACTLY one option, copied VERBATIM from the list.\n"
    "2) Output ONLY the diagnosis text. No explanation. No punctuation. No quotes."
)

SEED_FEW_SHOT = """Example 1:
Case: A 63-year-old woman presents with left serous otitis media for one month, xerophthalmia, and inability to close her left eye. Cranial nerve VI, VII, IX, X, and XII palsies are noted on the left side. MRI shows a large nasopharyngeal mass with skull base invasion.
Options: Adenoid cystic carcinoma, Chordoma, Nonkeratinising squamous cell carcinoma of the nasopharynx, Extramedullary plasmacytoma, Extranodal non-Hodgkin lymphoma
Answer: Nonkeratinising squamous cell carcinoma of the nasopharynx

Example 2:
Case: 49-year-old male with dysphagia and 3kg weight loss in 3 months. Barium study shows annular narrowing ring of the distal oesophagus. Endoscopy shows a thin membranous ring at the gastroesophageal junction.
Options: Carcinoma of the distal oesophagus, Annular peptic strictures, Schatzki ring, Distal oesophageal webs
Answer: Schatzki ring

Example 3:
Case: A 31-year-old female with a painless, slow-growing mass in the right abdominal wall. Palpable fixed and firm tumour in the right lateral abdominal wall. CT shows a well-defined, homogeneous, enhancing soft tissue mass arising from the internal oblique muscle.
Options: Fibrosarcoma, The definite histologic diagnosis was of dermoid-type fibromatosis., Lymphoma, Haematoma, Leiomyosarcoma
Answer: The definite histologic diagnosis was of dermoid-type fibromatosis."""

USER_TEMPLATE = (
    "Here are some example cases and their correct diagnoses:\n\n"
    "{few_shot}\n\n"
    "---\n\n"
    "Now select the diagnosis for this case:\n\n"
    "Case description:\n{case_text}\n\n"
    "Candidate diagnoses (choose ONE):\n{options_block}\n\n"
    "Return exactly one option from the list above, copied verbatim."
)

# ---------------------------------------------------------------------------
# Text normalization (mirrors hf_bench.py)
# ---------------------------------------------------------------------------


def norm_text(s: str) -> str:
    t = s or ""
    try:
        t = t.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[^\w\s-]", "", t)
    t = " ".join(t.strip().split())
    return t.lower()


# ---------------------------------------------------------------------------
# Answer extraction (mirrors hf_bench.py)
# ---------------------------------------------------------------------------

FINAL_CH_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)$", re.DOTALL)


def extract_final(text: str) -> str:
    if not text:
        return ""
    m = FINAL_CH_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def build_options_list(s: str) -> list[str]:
    opts = [o.strip() for o in (s or "").split(",") if o.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for o in opts:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def map_to_option(raw_answer: str, options: list[str]) -> tuple[str, str]:
    raw = (raw_answer or "").strip()
    if not raw or not options:
        return "", "no_match"
    if raw in options:
        return raw, "exact"
    norm2opt = {norm_text(o): o for o in options}
    nr = norm_text(raw)
    if nr in norm2opt:
        return norm2opt[nr], "normalized"
    cand = difflib.get_close_matches(raw, options, n=1, cutoff=0.8)
    if cand:
        return cand[0], "fuzzy"
    norm_opts = list(norm2opt.keys())
    candn = difflib.get_close_matches(nr, norm_opts, n=1, cutoff=0.9)
    if candn:
        return norm2opt[candn[0]], "fuzzy"
    return "", "no_match"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(csv_path: Path, id_col: str | None = None) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if id_col is None:
        if "case_id" in df.columns:
            id_col = "case_id"
        elif " " in df.columns:
            id_col = " "
    examples = []
    for _, row in df.iterrows():
        final_dx = row.get("FinalDiagnosis")
        if pd.isna(final_dx) or not str(final_dx).strip():
            continue
        desc = str(row.get("PostDescription") or row.get("OriginalDescription") or "").strip()
        ddx_list = str(row.get("DifferentialDiagnosisList", "")).strip()
        case_id = str(row[id_col]) if id_col and id_col in row.index else ""
        examples.append({
            "case_id": case_id,
            "description": desc,
            "differential_diagnosis_list": ddx_list,
            "final_diagnosis": str(final_dx).strip(),
        })
    return examples


# ---------------------------------------------------------------------------
# Task model caller — captures reasoning trace
# ---------------------------------------------------------------------------


def call_task_model(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """Call gpt-oss-120b. Returns (final_answer, reasoning_trace)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = litellm.completion(
                model=TASK_MODEL,
                messages=[
                    {"role": "system", "content": f"{system_prompt}\nReasoning: high."},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            msg = response.choices[0].message
            raw = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", "") or ""
            if not raw:
                raise RuntimeError("empty response from task model")
            return extract_final(raw), reasoning
        except Exception as e:
            if attempt >= MAX_RETRIES:
                return f"ERROR: {type(e).__name__}: {e}", ""
            time.sleep(BASE_BACKOFF * (2**attempt))
    return "", ""


# ---------------------------------------------------------------------------
# GEPA evaluator — two components: system_prompt + few_shot
# ---------------------------------------------------------------------------


def evaluator(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    system_prompt = candidate["system_prompt"]
    few_shot = candidate["few_shot"]
    options = build_options_list(example["differential_diagnosis_list"])
    options_block = "\n".join(f"- {o}" for o in options) if options else "- (no options provided)"
    user_prompt = USER_TEMPLATE.format(
        few_shot=few_shot,
        case_text=example["description"],
        options_block=options_block,
    )

    raw_answer, reasoning = call_task_model(system_prompt, user_prompt)
    mapped, match_type = map_to_option(raw_answer, options)

    gold = norm_text(example["final_diagnosis"])
    pred = norm_text(mapped)
    correct = int(gold == pred)

    # Rich side info with reasoning trace
    side_info: dict[str, Any] = {
        "Input": f"Case: {example['description'][:300]}...\nOptions: {', '.join(options)}",
        "Output": raw_answer,
        "Expected": example["final_diagnosis"],
        "match_type": match_type,
    }

    if correct:
        side_info["Feedback"] = (
            f"Correct. Model selected '{mapped}' (match: {match_type}).\n"
            f"Model reasoning: {reasoning[:500]}"
        )
    else:
        side_info["Feedback"] = (
            f"Incorrect.\n"
            f"  Model output: '{raw_answer}'\n"
            f"  Mapped to: '{mapped}' ({match_type})\n"
            f"  Correct answer: '{example['final_diagnosis']}'\n"
            f"  All options: {', '.join(options)}\n"
            f"\n  Model reasoning (WHY it chose wrong):\n  {reasoning[:800]}"
        )

    return float(correct), side_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Load and split data
    print(f"Loading data from {TRAIN_CSV}")
    all_train = load_dataset(TRAIN_CSV)
    rng = random.Random(SPLIT_SEED)
    shuffled = list(all_train)
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * VAL_FRACTION)
    valset = shuffled[:val_size]
    trainset = shuffled[val_size:]
    print(f"  Split: {len(trainset)} train / {len(valset)} val")

    testset = load_dataset(TEST_CSV)
    print(f"  Held-out test: {len(testset)} examples")

    # GEPA config
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=MAX_METRIC_CALLS,
            max_candidate_proposals=MAX_CANDIDATE_PROPOSALS,
            run_dir=str(RUN_DIR),
            cache_evaluation=True,
            cache_evaluation_storage="disk",
            candidate_selection_strategy="pareto",
            parallel=True,
            max_workers=8,
            seed=42,
            display_progress_bar=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=REFLECTION_LM,
            reflection_minibatch_size=REFLECTION_MINIBATCH_SIZE,
            module_selector="round_robin",  # alternate between system_prompt and few_shot
        ),
        merge=MergeConfig(max_merge_invocations=3),
    )

    print(f"\nGEPA Enhanced config:")
    print(f"  Strategy:         pareto")
    print(f"  Components:       system_prompt + few_shot (round_robin)")
    print(f"  Task model:       {TASK_MODEL} (Reasoning: high)")
    print(f"  Reflection LM:    {REFLECTION_LM}")
    print(f"  Max metric calls: {MAX_METRIC_CALLS}")
    print(f"  Minibatch size:   {REFLECTION_MINIBATCH_SIZE}")
    print(f"  Run dir:          {RUN_DIR}")

    print("\n=== Starting GEPA Enhanced ===\n")
    result = optimize_anything(
        seed_candidate={
            "system_prompt": SEED_SYSTEM_PROMPT,
            "few_shot": SEED_FEW_SHOT,
        },
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=(
            "Maximize diagnostic accuracy on radiology case MCQs. "
            "The model receives a system prompt, few-shot examples, and a case with candidate diagnoses. "
            "It must select exactly one diagnosis verbatim from the list. "
            "Optimize both the system prompt instructions AND the few-shot examples."
        ),
        background=(
            "Task model: gpt-oss-120b with high reasoning, a 120B-parameter on-device LLM. "
            "Two components are being optimized:\n"
            "1) system_prompt: Instructions for how to reason about radiology cases\n"
            "2) few_shot: Example cases with correct diagnoses shown before each query\n"
            "The model's internal reasoning trace is captured and included in feedback. "
            "When the model gets a case wrong, its reasoning shows WHY it chose incorrectly — "
            "use this to identify systematic reasoning errors and fix them via better instructions or examples. "
            "Common failure modes: confusing similar conditions, missing pathognomonic imaging signs, "
            "anchoring on the most common diagnosis instead of the best-fitting rare one, "
            "and paraphrasing instead of copying the answer verbatim."
        ),
        config=config,
    )

    # Report
    print("\n=== GEPA Enhanced complete ===\n")
    best = result.best_candidate
    print(f"Best system prompt:\n{'-'*60}")
    print(best["system_prompt"])
    print(f"{'-'*60}")
    print(f"\nBest few-shot:\n{'-'*60}")
    print(best["few_shot"])
    print(f"{'-'*60}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "best_system_prompt.txt").write_text(best["system_prompt"])
    (RUN_DIR / "best_few_shot.txt").write_text(best["few_shot"])

    # Held-out test evaluation
    testset = load_dataset(TEST_CSV)
    print(f"\n=== Evaluating on held-out test ({len(testset)} examples) ===\n")
    opt_correct = 0
    for i, ex in enumerate(testset):
        score, _ = evaluator(best, ex)
        opt_correct += int(score)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(testset)}: {opt_correct}/{i+1} ({opt_correct/(i+1):.3f})")
    opt_acc = opt_correct / len(testset)
    print(f"\nHeld-out test accuracy: {opt_correct}/{len(testset)} ({opt_acc:.4f})")
    (RUN_DIR / "results_summary.txt").write_text(f"Test: {opt_acc:.4f} ({opt_correct}/{len(testset)})\n")
    print(f"Saved to: {RUN_DIR / 'results_summary.txt'}")


if __name__ == "__main__":
    main()
