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
from torch.backends.cuda import sdp_kernel
from unsloth import FastLanguageModel
import argparse


# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "openai/gpt-oss-20b"
DEFAULT_MAX_SEQ_LEN = 4096

DEFAULT_CUDA_VISIBLE_DEVICES = "1"
DEFAULT_ATTENTION_BACKEND = "eager"

DEFAULT_DATA_DIR = "/home/alhusain/scratch/ondevice-llm"
DEFAULT_VAL_CSV = f"{DEFAULT_DATA_DIR}/ophthalmology_mcq.csv"
DEFAULT_OUT_CSV = f"{DEFAULT_DATA_DIR}/final_results/oph_mcq_5beams.csv"

DEFAULT_NUM_BEAMS = 5
DEFAULT_DIVERSITY_PENALTY = 0.5
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_NO_REPEAT_NGRAM_SIZE = 0
DEFAULT_MAX_NEW_TOKENS = 3500

SHORT_INSTR = (
    "You are an ophthalmology multiple-choice assistant.\n"
    "Output ONLY the correct answer letters (A–K), concatenated without spaces or punctuation.\n"
    "Examples: A, BD, ACE.\n"
    "Do not output any words, explanations, or reasoning.\n"
    "Your response MUST be letters only."
)

ASSISTANTFINAL_TAG = re.compile(r"assistant\s*final", re.I)


# --------------------------------------------------------------------------------------
# Environment / model setup
# --------------------------------------------------------------------------------------
def setup_environment(cuda_visible_devices: str, attention_backend: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["HF_PYTORCH_ATTENTION_BACKEND"] = attention_backend
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


def load_base_model(base_model: str, max_seq_len: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        fast_inference=False,
        attn_implementation="eager",
        float8_kv_cache=True,
    )
    model.eval()

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    print("Ready. Base model only (no LoRA).")
    print("Device:", device, "| Max seq len:", max_seq_len)

    return model, tokenizer


# --------------------------------------------------------------------------------------
# Prompt helpers
# --------------------------------------------------------------------------------------
def make_msgs_from_row(row: pd.Series):
    question = str(row.get("Question", "")).strip()
    user = f"{SHORT_INSTR}\n\nQuestion: {question}\n"
    return [
        {"role": "system", "content": "You are an expert ophthalmology assistant."},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------------------
# Parsers (letters only)
# --------------------------------------------------------------------------------------
def _normalize_choice_letters_only(text: str) -> str:
    s = (text or "").upper()
    letters = re.findall(r"[A-K]", s)
    if not letters:
        return ""
    ordered = [ch for ch in "ABCDEFGHIJK" if ch in letters]
    seen, cleaned = set(), []
    for ch in ordered:
        if ch not in seen:
            seen.add(ch)
            cleaned.append(ch)
    return "".join(cleaned)


def parse_after_assistantfinal(text: str) -> str:
    if not text:
        return ""
    m = ASSISTANTFINAL_TAG.search(text)
    if not m:
        return ""
    tail = text[m.end():]
    tail = re.sub(r"^[\s:>\]\)\-–—_]*", "", tail)
    m2 = re.search(r"([A-Ka-k]{1,10})", tail)
    if not m2:
        return ""
    return _normalize_choice_letters_only(m2.group(1))


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
def gen_cot_beams(
    model,
    tokenizer,
    msgs,
    effort: str = "medium",
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
# Main evaluation
# --------------------------------------------------------------------------------------
def run_ophthalmology_mcq_inference(
    model,
    tokenizer,
    val_csv: Path,
    out_csv: Path,
    num_beams: int,
    diversity_penalty: float,
    length_penalty: float,
    no_repeat_ngram_size: int,
    max_new_tokens: int,
):
    df = pd.read_csv(val_csv)
    print(df.shape)
    df = df.iloc[0:].copy()

    model.eval()

    try:
        model.generation_config.trust_remote_code = True
    except Exception:
        pass

    final_answers = []
    beam_labels_col, all_beams_text = [], []

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
        )

        per_beam_labels = [parse_after_assistantfinal(g) for g in gens]

        counts = Counter([lab for lab in per_beam_labels if lab])
        if counts:
            max_count = max(counts.values())
            tied_labels = [lab for lab, c in counts.items() if c == max_count]
            if len(tied_labels) == 1:
                chosen_label = tied_labels[0]
            else:
                first_idx = {
                    lab: next(
                        i for i, lab_i in enumerate(per_beam_labels) if lab_i == lab
                    )
                    for lab in tied_labels
                }
                chosen_label = min(tied_labels, key=lambda lab: first_idx[lab])
        else:
            chosen_label = next((lab for lab in per_beam_labels if lab), "")

        final_answers.append(chosen_label)
        beam_labels_col.append(json.dumps(per_beam_labels, ensure_ascii=False))
        all_beams_text.append(" ||| ".join(gens))

    df_out = df.copy()
    df_out["final_answer"] = final_answers
    df_out["beam_final_labels"] = beam_labels_col
    df_out["raw_generations"] = all_beams_text

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print("Saved:", str(out_csv))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ophthalmology MCQ evaluation with GPT-OSS-20B using diverse beam search."
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model name for Unsloth FastLanguageModel.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=DEFAULT_CUDA_VISIBLE_DEVICES,
        help="CUDA_VISIBLE_DEVICES value.",
    )
    parser.add_argument(
        "--attention-backend",
        default=DEFAULT_ATTENTION_BACKEND,
        help="HF_PYTORCH_ATTENTION_BACKEND value.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path(DEFAULT_VAL_CSV),
        help="Path to ophthalmology MCQ CSV.",
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
        default=DEFAULT_DIVERSITY_PENALTY,
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
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens to generate.",
    )

    args = parser.parse_args()

    setup_environment(args.cuda_visible_devices, args.attention_backend)
    model, tokenizer = load_base_model(args.base_model, args.max_seq_len)

    run_ophthalmology_mcq_inference(
        model=model,
        tokenizer=tokenizer,
        val_csv=args.val_csv,
        out_csv=args.out_csv,
        num_beams=args.num_beams,
        diversity_penalty=args.diversity_penalty,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
