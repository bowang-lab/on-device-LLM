#!/usr/bin/env python3
"""
CoT diverse-beam inference + external rescoring using Hugging Face Transformers
with the official "openai/gpt-oss-120b" model (no Unsloth). Optionally loads a
LoRA adapter via PEFT if LORA_PATH is provided.

Outputs a CSV with columns:
  - reasoning                (chosen beam's <analysis> ... </analysis>)
  - final_answer             (chosen label from DifferentialDiagnosisList)
  - beam_final_labels        (JSON list of all beams' <final> labels)
  - beam_sequence_scores     (JSON list of external mean log-prob scores)
  - raw_generations          (all full generations joined by " ||| ")

Notes:
  - This uses group beam search with diversity_penalty for deterministic diversity.
  - External rescoring recomputes p(continuation | prompt) using mean token log-prob.
  - If ≥AGREE_THRESHOLD of beams agree on the same <final>, that label is chosen;
    otherwise the single highest-score beam is chosen.
"""

import os
import re
import json
import argparse
from collections import Counter
from typing import List, Tuple
from tqdm import tqdm

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# PEFT is optional; keep compatibility with an external LoRA adapter
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# -------------------------------
# Config
# -------------------------------
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b")
LORA_PATH = os.environ.get("LORA_PATH", "")  # optional
DATA_DIR = os.environ.get("DATA_DIR", ".")
IN_CSV = os.environ.get("IN_CSV", os.path.join(DATA_DIR, "eurorad_val.csv"))
OUT_CSV = os.environ.get("OUT_CSV", os.path.join(DATA_DIR, "val_preds_newscored_18beams_hf.csv"))

# beam search params
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", 9))
NUM_BEAM_GROUPS = int(os.environ.get("NUM_BEAM_GROUPS", NUM_BEAMS))
DIVERSITY_PENALTY = float(os.environ.get("DIVERSITY_PENALTY", 0.5))
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", 1.0))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 768))

# selection params
AGREE_THRESHOLD = float(os.environ.get("AGREE_THRESHOLD", 0.5))

# model/runtime params
ATTN_IMPL = os.environ.get("ATTN_IMPL", "eager")   # "eager" is safest
DTYPE = os.environ.get("DTYPE", "bfloat16")         # "bfloat16" or "float16"
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")   # e.g., "auto"

# If you want 4-bit load and your stack supports it, set LOAD_4BIT=1
LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"

# -------------------------------
# Prompting
# -------------------------------
SYS_PROMPT = (
    "You are a careful clinical reasoning assistant for radiology cases. "
    "Given a case description and a finite list of possible diagnoses, "
    "think step by step, then return a single final answer that matches "
    "exactly one label from the provided candidate list."
)

USER_TEMPLATE = (
    "Case description:\n"  # PostDescription
    "{post}\n\n"
    "Candidate diagnoses (copy exactly one):\n"
    "{cands}\n\n"
    "Respond in the following strict XML format:\n"
    "<analysis>YOUR DETAILED REASONING HERE</analysis>\n"
    "<final>ONE LABEL COPIED VERBATIM FROM THE LIST</final>\n"
)

# -------------------------------
# Utilities
# -------------------------------
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL | re.IGNORECASE)
ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)


def parse_final_only(text: str) -> str:
    """Extract final label from the *continuation* text (not the prompt). Guard against template placeholders."""
    if not text:
        return ""
    m = FINAL_RE.search(text)
    if m:
        val = m.group(1).strip()
    else:
        # fallback: last non-empty line from continuation
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        val = lines[-1] if lines else ""
    up = val.upper()
    if up.startswith("ONE LABEL COPIED VERBATIM") or up.startswith("LETTERS A"):
        return ""  # ignore template placeholders
    return val


def parse_analysis(text: str) -> str:
    """Extract analysis text from the *continuation*; ignore placeholder."""
    if not text:
        return ""
    m = ANALYSIS_RE.search(text)
    val = m.group(1).strip() if m else text.strip()
    if val.upper().startswith("YOUR DETAILED REASONING HERE"):
        return ""
    return val


def build_prompt(post: str, candidates: List[str]) -> str:
    cands = "\n".join(candidates)
    return f"{SYS_PROMPT}\n\n" + USER_TEMPLATE.format(post=post.strip(), cands=cands)


# -------------------------------
# HF model loading
# -------------------------------

def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    quant_kwargs = {}
    if LOAD_4BIT:
        # Requires bitsandbytes and a compatible install
        from transformers import BitsAndBytesConfig
        quant_kwargs = {
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
            "device_map": DEVICE_MAP,
        }

    dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # ensure left padding for generation/beam search

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=ATTN_IMPL,
        torch_dtype=dtype,
        device_map=DEVICE_MAP,
        trust_remote_code=TRUST_REMOTE_CODE,
        **quant_kwargs,
    )

    if LORA_PATH:
        if not _HAS_PEFT:
            raise RuntimeError("PEFT not installed but LORA_PATH provided")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model.eval()

    return model, tok


# -------------------------------
# Generation & rescoring
# -------------------------------

def gen_cot_beams(model, tok, prompt: str) -> Tuple[List[str], torch.LongTensor, torch.LongTensor, List[torch.LongTensor]]:
    """Return (decoded_generations, prompt_ids, prompt_attn_mask, list_of_continuation_ids)."""
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attn_mask = inputs["attention_mask"].to(model.device)

    gen_cfg = GenerationConfig(
        do_sample=False,
        num_beams=NUM_BEAMS,
        num_beam_groups=NUM_BEAM_GROUPS,
        diversity_penalty=DIVERSITY_PENALTY,
        length_penalty=LENGTH_PENALTY,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            **gen_cfg.to_dict(),
        )

    # out.sequences shape: [num_beams, prompt_len + gen_len] (since input batch size = 1)
    sequences = out.sequences
    prompt_len = input_ids.shape[1]

    # Split into per-beam continuations (strip the prompt prefix)
    cont_ids_list = []
    decoded = []
    for i in range(sequences.size(0)):
        full = sequences[i]
        cont = full[prompt_len:]
        cont_ids_list.append(cont)
        # Decode ONLY the continuation to avoid parsing placeholders from the prompt
        decoded.append(tok.decode(cont, skip_special_tokens=True))

    return decoded, input_ids[0], attn_mask[0], cont_ids_list


def score_continuation_given_ids(model, tok, prompt_ids: torch.LongTensor, cont_ids: torch.LongTensor) -> float:
    """Mean token log-prob of continuation given prompt."""
    device = model.device
    full = torch.cat([prompt_ids, cont_ids]).unsqueeze(0).to(device)
    attn = torch.ones_like(full, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=full, attention_mask=attn).logits  # [1, L, V]
    # shift
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, L-1, V]
    targets = full[:, 1:]  # [1, L-1]

    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

    # Only count continuation positions
    prompt_len = prompt_ids.numel()
    cont_mask = torch.zeros_like(token_lp, dtype=torch.bool)
    cont_mask[:, prompt_len-1:] = True  # positions whose prediction consumes continuation

    cont_token_lp = token_lp[cont_mask]
    if cont_token_lp.numel() == 0:
        return float("-inf")
    return float(cont_token_lp.mean().item())


# -------------------------------
# Main utilities
# -------------------------------

def print_config(args):
    cfg = {
        "in_csv": args.in_csv,
        "out_csv": args.out_csv,
        "model": args.model,
        "start_row": args.start_row,
        "NUM_BEAMS": NUM_BEAMS,
        "NUM_BEAM_GROUPS": NUM_BEAM_GROUPS,
        "DIVERSITY_PENALTY": DIVERSITY_PENALTY,
        "LENGTH_PENALTY": LENGTH_PENALTY,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "AGREE_THRESHOLD": AGREE_THRESHOLD,
        "ATTN_IMPL": ATTN_IMPL,
        "DTYPE": DTYPE,
        "TRUST_REMOTE_CODE": TRUST_REMOTE_CODE,
        "DEVICE_MAP": DEVICE_MAP,
        "LOAD_4BIT": LOAD_4BIT,
        "LORA_PATH": LORA_PATH,
    }
    print("=== Run Config ===")
    print(json.dumps(cfg, indent=2))

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default=IN_CSV)
    parser.add_argument("--out_csv", default=OUT_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start_row", type=int, default=0, help="0-based start row index")
    parser.add_argument("--stream", action="store_true", help="Write/append rows to CSV as they are processed")
    parser.add_argument("--save_every_n", type=int, default=1, help="Stream-write every N rows (only with --stream)")
    args = parser.parse_args()

    print_config(args)
    model, tok = load_model_and_tokenizer(args.model)
    model.eval()

    df = pd.read_csv(args.in_csv)

    # Prepare streaming output
    written_header = False
    if args.stream:
        # start fresh
        try:
            if os.path.exists(args.out_csv):
                os.remove(args.out_csv)
        except Exception:
            pass

    if args.start_row > 0:
        df = df.iloc[args.start_row:].reset_index(drop=True)

    required_cols = ["PostDescription", "DifferentialDiagnosisList"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    final_answers: List[str] = []
    top_reasonings: List[str] = []
    beam_labels_col: List[str] = []
    beam_scores_col: List[str] = []
    all_beams_text_col: List[str] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="CoT beams"):
        post = str(row["PostDescription"]) if not pd.isna(row["PostDescription"]) else ""
        cand_raw = str(row["DifferentialDiagnosisList"]) if not pd.isna(row["DifferentialDiagnosisList"]) else ""
        # candidates can be comma- or newline-separated
        candidates = [c.strip() for c in re.split(r"\n|,", cand_raw) if c.strip()]
        if not post or not candidates:
            final_answers.append("")
            top_reasonings.append("")
            beam_labels_col.append(json.dumps([]))
            beam_scores_col.append(json.dumps([]))
            all_beams_text_col.append("")
            continue

        prompt = build_prompt(post, candidates)
        gens, prompt_ids, prompt_mask, cont_ids_list = gen_cot_beams(model, tok, prompt)

        # external rescoring per beam
        scores = []
        per_beam_labels = []
        for cont_ids, continuation_text in zip(cont_ids_list, gens):
            s = score_continuation_given_ids(model, tok, prompt_ids, cont_ids)
            scores.append(s)
            per_beam_labels.append(parse_final_only(continuation_text))

        # consensus vs argmax
        counts = Counter([lab for lab in per_beam_labels if lab])
        if counts:
            top_lab, top_count = counts.most_common(1)[0]
            ratio = top_count / max(1, len(per_beam_labels))
        else:
            top_lab, ratio = "", 0.0

        chosen_label = ""
        chosen_reason = ""
        if ratio >= AGREE_THRESHOLD and top_lab:
            chosen_label = top_lab
            # take reasoning from the earliest beam that predicted it
            for continuation_text, lab in zip(gens, per_beam_labels):
                if lab == top_lab:
                    chosen_reason = parse_analysis(continuation_text)
                    break
        else:
            # pick highest-score beam
            if scores:
                best_i = max(range(len(scores)), key=lambda i: scores[i])
                chosen_label = per_beam_labels[best_i] if per_beam_labels else ""
                chosen_reason = parse_analysis(gens[best_i]) if gens else ""

        final_answers.append(chosen_label)
        top_reasonings.append(chosen_reason)
        beam_labels_col.append(json.dumps(per_beam_labels, ensure_ascii=False))
        beam_scores_col.append(json.dumps(scores))
        all_beams_text_col.append(" ||| ".join(gens))

        # Stream write this row if enabled
        if args.stream and ( (idx + 1) % args.save_every_n == 0 or (idx + 1) == len(df) ):
            out_row = row.to_dict()
            out_row.update({
                "reasoning": chosen_reason,
                "final_answer": chosen_label,
                "beam_final_labels": json.dumps(per_beam_labels, ensure_ascii=False),
                "beam_sequence_scores": json.dumps(scores),
                "raw_generations": " ||| ".join(gens),
            })
            out_df = pd.DataFrame([out_row])
            # Ensure directory exists
            os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
            with open(args.out_csv, "a", buffering=1) as f:
                out_df.to_csv(f, header=not written_header, index=False)
                try:
                    f.flush(); os.fsync(f.fileno())
                except Exception:
                    pass
            written_header = True

    if args.stream:
        print("Streaming writes enabled; CSV updated during the run at:", args.out_csv)
    else:
        # write once at the end
        df_out = df.copy()
        df_out["reasoning"] = top_reasonings
        df_out["final_answer"] = final_answers
        df_out["beam_final_labels"] = beam_labels_col
        df_out["beam_sequence_scores"] = beam_scores_col
        df_out["raw_generations"] = all_beams_text_col

        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df_out.to_csv(args.out_csv, index=False)
        print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
