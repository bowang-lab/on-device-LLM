#!/usr/bin/env python3
"""
GEPA optimize_anything on Eurorad — Run 3: pareto strategy, high reasoning.

Seeds from the manual prompt (not Run 1's best) with high reasoning and pareto
strategy for broader exploration.

Uses AWS Bedrock for both models:
  - Task model:      gpt-oss-120b (openai.gpt-oss-120b-1:0)
  - Reflection LM:   Claude Opus 4.6 (anthropic.claude-opus-4-6-v1)
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
RUN_DIR = Path(__file__).resolve().parent / "runs" / "eurorad_oss120b_r3"

VAL_FRACTION = 0.2
SPLIT_SEED = 42  # same split as Run 1

MAX_METRIC_CALLS = 5000
MAX_CANDIDATE_PROPOSALS = 50
REFLECTION_MINIBATCH_SIZE = 5
MAX_RETRIES = 3
BASE_BACKOFF = 2.0

USER_TEMPLATE = (
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
# Task model caller
# ---------------------------------------------------------------------------


def call_task_model(system_prompt: str, user_prompt: str) -> str:
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
            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                raise RuntimeError("empty response from task model")
            return extract_final(raw)
        except Exception as e:
            if attempt >= MAX_RETRIES:
                return f"ERROR: {type(e).__name__}: {e}"
            time.sleep(BASE_BACKOFF * (2**attempt))
    return ""


# ---------------------------------------------------------------------------
# GEPA evaluator
# ---------------------------------------------------------------------------


def evaluator(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    system_prompt = candidate["system_prompt"]
    options = build_options_list(example["differential_diagnosis_list"])
    options_block = "\n".join(f"- {o}" for o in options) if options else "- (no options provided)"
    user_prompt = USER_TEMPLATE.format(case_text=example["description"], options_block=options_block)

    raw_answer = call_task_model(system_prompt, user_prompt)
    mapped, match_type = map_to_option(raw_answer, options)

    gold = norm_text(example["final_diagnosis"])
    pred = norm_text(mapped)
    correct = int(gold == pred)

    side_info: dict[str, Any] = {
        "Input": f"Case: {example['description'][:300]}...\nOptions: {', '.join(options)}",
        "Output": raw_answer,
        "Expected": example["final_diagnosis"],
        "match_type": match_type,
    }
    if correct:
        side_info["Feedback"] = f"Correct. Model selected '{mapped}' (match: {match_type})."
    else:
        side_info["Feedback"] = (
            f"Incorrect.\n"
            f"  Model output: '{raw_answer}'\n"
            f"  Mapped to: '{mapped}' ({match_type})\n"
            f"  Correct answer: '{example['final_diagnosis']}'\n"
            f"  All options: {', '.join(options)}"
        )
    return float(correct), side_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Load seed prompt from Run 1
    seed_prompt = (
        "You are a careful radiology diagnosis selector.\n"
        "Given a case description and a finite list of candidate diagnoses, "
        "select the single most likely final diagnosis FROM THE LIST.\n"
        "Response rules:\n"
        "1) Output EXACTLY one option, copied VERBATIM from the list.\n"
        "2) Output ONLY the diagnosis text. No explanation. No punctuation. No quotes."
    )
    print(f"Seeding from manual prompt ({len(seed_prompt)} chars)")

    # Load and split data (same split as Run 1)
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

    # GEPA config — pareto strategy
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
            module_selector="all",
        ),
        merge=MergeConfig(max_merge_invocations=3),
    )

    print(f"\nGEPA Run 2 config:")
    print(f"  Strategy:         pareto (exploratory)")
    print(f"  Task model:       {TASK_MODEL}")
    print(f"  Reflection LM:    {REFLECTION_LM}")
    print(f"  Max metric calls: {MAX_METRIC_CALLS}")
    print(f"  Max proposals:    {MAX_CANDIDATE_PROPOSALS}")
    print(f"  Minibatch size:   {REFLECTION_MINIBATCH_SIZE}")
    print(f"  Run dir:          {RUN_DIR}")

    print("\n=== Starting GEPA Run 2 ===\n")
    result = optimize_anything(
        seed_candidate={"system_prompt": seed_prompt},
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=(
            "Maximize diagnostic accuracy on radiology case MCQs. "
            "The model receives a clinical case description and a list of candidate diagnoses, "
            "and must select exactly one diagnosis verbatim from the list."
        ),
        background=(
            "Task model: gpt-oss-120b, a 120B-parameter on-device LLM for clinical decision support. "
            "The model outputs a single diagnosis string. Evaluation is exact match after text normalization. "
            "Common failure modes: selecting a diagnosis not on the list, paraphrasing instead of copying verbatim, "
            "outputting explanations alongside the answer, and confusing similar-sounding conditions. "
            "The system prompt should guide the model to reason carefully about the clinical presentation "
            "before selecting, while keeping output format strictly verbatim."
        ),
        config=config,
    )

    # Report
    print("\n=== GEPA Run 3 complete ===\n")
    best_prompt = result.best_candidate["system_prompt"]
    print(f"Best system prompt:\n{'-'*60}")
    print(best_prompt)
    print(f"{'-'*60}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "best_system_prompt.txt").write_text(best_prompt)

    # Held-out test evaluation
    testset = load_dataset(TEST_CSV)
    print(f"\n=== Evaluating on held-out test ({len(testset)} examples) ===\n")
    opt_correct = 0
    for i, ex in enumerate(testset):
        score, _ = evaluator({"system_prompt": best_prompt}, ex)
        opt_correct += int(score)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(testset)}: {opt_correct}/{i+1} ({opt_correct/(i+1):.3f})")
    opt_acc = opt_correct / len(testset)
    print(f"\nHeld-out test accuracy: {opt_correct}/{len(testset)} ({opt_acc:.4f})")
    (RUN_DIR / "results_summary.txt").write_text(f"Test: {opt_acc:.4f} ({opt_correct}/{len(testset)})\n")
    print(f"Saved to: {RUN_DIR / 'results_summary.txt'}")


if __name__ == "__main__":
    main()
