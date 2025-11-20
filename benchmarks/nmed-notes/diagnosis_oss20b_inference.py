#!/usr/bin/env python
import os
import re
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.backends.cuda import sdp_kernel
import argparse


# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------
DEFAULT_MODEL_DIR = os.path.expandvars(
    os.getenv("OSS20B_DIR", "$SCRATCH/models/gpt-oss-20b")
)

DEFAULT_DATA_DIR = os.path.expandvars("$PROJECT/data")
DEFAULT_VAL_CSV = os.path.join(DEFAULT_DATA_DIR, "nmed/nmed_diagnosis.csv")
DEFAULT_OUT_CSV = os.path.join(
    DEFAULT_DATA_DIR,
    "results/nmed_diagnosis_5beams_try8_low_effort_reasoning_v4.csv",
)

DEFAULT_NUM_BEAMS = 5
DEFAULT_DIV_PENALTY = 0.5
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_NO_REPEAT_NGRAM_SIZE = 0
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_NEW_TOKENS = 3000

DEFAULT_ATTENTION_BACKEND = "eager"

EVAL_PROMPT = """You are asked to evaluate the quality of a model’s diagnostic output using the following rubric:

**Scoring Rubric (Likert scale 1–5):**
1. Most relevant options not mentioned.
2. Some or many relevant options not mentioned.
3. Most relevant options mentioned.
4. Most relevant options mentioned.
5. All relevant options mentioned.

**Instruction:**  
Given the following task description, the true disease, and the model output, assign a single integer score from 1 to 5 according to the rubric. Half-point scores (e.g., 1.5, 2.5, 3.5, 4.5) are allowed if the quality falls between two rubric levels.
Output **only the score**, with no explanation or justification.
"""

ASSISTANTFINAL_TAG = re.compile(r"assistant\s*final", re.I)
SCORE_AFTER_TAG = re.compile(r"([0-5](?:\.[05])?)")

# Global so it matches original behavior inside gen_cot_beams
REASONING_EFFORT = DEFAULT_REASONING_EFFORT


# --------------------------------------------------------------------------------------
# Environment / model setup
# --------------------------------------------------------------------------------------
def setup_environment(attention_backend: str) -> None:
    os.environ["HF_PYTORCH_ATTENTION_BACKEND"] = attention_backend
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    os.environ.setdefault("HF_HOME", os.path.expandvars("$SCRATCH/hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_HOME"])
    os.environ.setdefault("TORCH_HOME", os.path.expandvars("$SCRATCH/torch_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _build_max_memory(reserve_gib: int = 2):
    assert torch.cuda.is_available(), "CUDA not available."
    n = torch.cuda.device_count()
    assert n >= 3, f"Need ≥3 GPUs visible; found {n}"
    max_mem = {}
    for i in range(n):
        total_mb = torch.cuda.get_device_properties(i).total_memory // (1024**2)
        target_gib = max(1, (total_mb // 1024) - reserve_gib)
        max_mem[i] = f"{target_gib}GiB"
    return max_mem


def load_model(model_dir: str):
    model_dir = os.path.expandvars(model_dir)
    assert os.path.isabs(model_dir), f"LOCAL_DIR is not an absolute path: {model_dir}"
    assert os.path.isdir(model_dir), f"Model dir not found: {model_dir}"

    max_memory = _build_max_memory(reserve_gib=2)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Ready. Base model only (no LoRA).")
    print("Device map:", getattr(model, "hf_device_map", None))
    print("Visible GPUs:", torch.cuda.device_count())
    print("Max per-GPU memory:", max_memory)
    print("Model dir:", model_dir)

    return model, tokenizer


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", on_bad_lines="skip")


# --------------------------------------------------------------------------------------
# Prompt helpers
# --------------------------------------------------------------------------------------
def make_msgs_from_row(row: pd.Series):
    post = str(row.get("Disease_description", "")).strip()
    user = f"{EVAL_PROMPT}\n\nInputs: {post}\n"
    return [
        {
            "role": "system",
            "content": "You are an expert clinical diagnostic assistant.",
        },
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------
def parse_score_after_assistantfinal(text: str):
    if not text:
        return None
    m = ASSISTANTFINAL_TAG.search(text)
    tail = text[m.end():] if m else text
    tail = re.sub(r"^[\s:>\]\)\-–—_]*", "", tail)
    m2 = SCORE_AFTER_TAG.search(tail)
    if not m2:
        return None
    try:
        val = float(m2.group(1))
    except Exception:
        return None
    return max(0.0, min(5.0, val))


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
def gen_cot_beams(
    model,
    tokenizer,
    msgs,
    effort: str,
    max_new_tokens: int = 8,
    beams: int = 3,
    diversity_penalty: float = 0.5,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    enc = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=REASONING_EFFORT,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,
            num_beams=beams,
            num_beam_groups=beams,
            num_return_sequences=beams,
            diversity_penalty=diversity_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
            output_scores=False,
        )

    prompt_len = enc["input_ids"].shape[1]
    gens, cont_ids_list = [], []
    for i in range(out.size(0)):
        cont_ids = out[i, prompt_len:]
        cont_ids_list.append(cont_ids.unsqueeze(0))
        gens.append(tokenizer.decode(cont_ids, skip_special_tokens=True).strip())
    return gens, enc, cont_ids_list


# --------------------------------------------------------------------------------------
# Main inference loop
# --------------------------------------------------------------------------------------
def run_nmed_diagnosis_inference(
    model,
    tokenizer,
    val_csv: Path,
    out_csv: Path,
    num_beams: int,
    diversity_penalty: float,
    length_penalty: float,
    no_repeat_ngram_size: int,
    checkpoint_every: int,
    reasoning_effort: str,
    max_new_tokens: int,
):
    global REASONING_EFFORT
    REASONING_EFFORT = reasoning_effort

    df = read_csv_robust(val_csv)
    print(df.shape)
    df = df.iloc[0:].copy()

    model.eval()
    try:
        model.generation_config.trust_remote_code = True
    except Exception:
        pass

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    final_answers = []
    beam_labels_col, all_beams_text, beam_parsed_scores_col = [], [], []
    top_reasonings = []

    def _write_csv_atomic(frame: pd.DataFrame, path: Path):
        tmp = str(path) + ".tmp"
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def _save_checkpoint(k: int):
        part = df.iloc[:k].copy()
        part["reasoning"] = top_reasonings[:k]
        part["final_answer"] = final_answers[:k]
        part["beam_parsed_scores"] = beam_parsed_scores_col[:k]
        part["raw_generations"] = all_beams_text[:k]
        _write_csv_atomic(part, out_csv)
        print(f"[CKPT] Saved {k} rows -> {out_csv}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        msgs = make_msgs_from_row(row)

        gens, _, _ = gen_cot_beams(
            model=model,
            tokenizer=tokenizer,
            msgs=msgs,
            max_new_tokens=max_new_tokens,
            beams=num_beams,
            diversity_penalty=diversity_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            effort=reasoning_effort,
        )

        per_beam_parsed_scores = [parse_score_after_assistantfinal(g) for g in gens]

        counts = Counter([sc for sc in per_beam_parsed_scores if sc is not None])

        if counts:
            max_count = max(counts.values())
            tied_scores = [sc for sc, c in counts.items() if c == max_count]

            if len(tied_scores) == 1:
                chosen_score = tied_scores[0]
            else:
                first_idx = {
                    sc: next(
                        i for i, s in enumerate(per_beam_parsed_scores) if s == sc
                    )
                    for sc in tied_scores
                }
                chosen_score = min(tied_scores, key=lambda sc: first_idx[sc])
        else:
            chosen_score = next(
                (sc for sc in per_beam_parsed_scores if sc is not None), None
            )

        top_reasonings.append("")
        final_answers.append("" if chosen_score is None else chosen_score)
        beam_parsed_scores_col.append(
            json.dumps(per_beam_parsed_scores, ensure_ascii=False)
        )
        all_beams_text.append(" ||| ".join(gens))

        k = len(final_answers)
        if k % checkpoint_every == 0:
            _save_checkpoint(k)

    _save_checkpoint(len(df))
    print("Saved:", out_csv)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NMED diagnosis scoring with GPT-OSS-20B using diverse beam search."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to local GPT-OSS-20B model directory.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path(DEFAULT_VAL_CSV),
        help="Path to NMED diagnosis CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(DEFAULT_OUT_CSV),
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=DEFAULT_NUM_BEAMS,
        help="Number of beams.",
    )
    parser.add_argument(
        "--diversity-penalty",
        type=float,
        default=DEFAULT_DIV_PENALTY,
        help="Diversity penalty.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=DEFAULT_LENGTH_PENALTY,
        help="Length penalty.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=DEFAULT_NO_REPEAT_NGRAM_SIZE,
        help="No repeat n-gram size.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Checkpoint interval (number of rows).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        help="Reasoning effort level passed to chat template.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=DEFAULT_ATTENTION_BACKEND,
        help="HF_PYTORCH_ATTENTION_BACKEND value.",
    )

    args = parser.parse_args()

    setup_environment(args.attention_backend)
    model, tokenizer = load_model(args.model_dir)

    run_nmed_diagnosis_inference(
        model=model,
        tokenizer=tokenizer,
        val_csv=args.val_csv,
        out_csv=args.out_csv,
        num_beams=args.num_beams,
        diversity_penalty=args.diversity_penalty,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        checkpoint_every=args.checkpoint_every,
        reasoning_effort=args.reasoning_effort,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
