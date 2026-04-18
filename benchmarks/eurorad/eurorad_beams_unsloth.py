#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Eurorad evaluation via Unsloth (no LoRA, no quantization), loading from a VALID local HF snapshot.
- Verifies shards exist before choosing a snapshot (avoids FileNotFoundError).
- Forces eager attention, left-padding, pad_token=eos.
- Deterministic diverse beam search (same decoding as your HF script).
- Appends rows to CSV; supports --resume.
- Prompts the model to output exactly 2 blocks: <analysis>...</analysis> then <final>...</final>.
- <final> MUST be exactly one option copied verbatim from the provided Differential diagnoses list.
- Extracts the final diagnosis using <final>...</final> (with fallbacks).

Input CSV columns:
  case_id, PostDescription, DifferentialDiagnosisList (comma-separated), FinalDiagnosis
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

# --- Eager attention (match HF script) ---
os.environ.setdefault("HF_PYTORCH_ATTENTION_BACKEND", "eager")
try:
    from torch.nn.attention import sdpa_kernel
    sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

from unsloth import FastLanguageModel


# ------------------------- Cache resolution (validated) -------------------------
def _roots():
    roots = []
    hf_home = os.environ.get("HF_HOME")
    hub_env = os.environ.get("HF_HUB_CACHE")
    tf_cache = os.environ.get("TRANSFORMERS_CACHE")

    if hub_env: roots.append(hub_env)
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
        roots.append(os.path.join(hf_home, "transformers"))
    if tf_cache: roots.append(tf_cache)
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))

    uniq = []
    for r in roots:
        if r and os.path.isdir(r) and r not in uniq:
            uniq.append(r)
    return uniq


def _snapshot_globs(repo_id: str):
    if "/" in repo_id:
        org, name = repo_id.split("/", 1)
    else:
        org, name = None, repo_id
    pats = []
    if org and name:
        pats.append(f"models--{org}--{name}/snapshots/*")
    if name:
        pats.append(f"models--unsloth--{name}/snapshots/*")
    return pats


def _parse_index_json(folder: Path):
    idx = folder / "model.safetensors.index.json"
    if not idx.exists():
        return None
    try:
        with idx.open("r", encoding="utf-8") as f:
            j = json.load(f)
        files = set(j.get("weight_map", {}).values())
        return {"index": idx, "files": sorted(files)}
    except Exception:
        return None


def _snapshot_is_valid(folder: Path) -> tuple[bool, int, list]:
    """
    Return (valid?, shard_count, missing_files_list).
    Valid if index.json exists AND all shard files exist; or if any *.safetensors present and no index.
    Prefer indexed snapshots; fall back to heuristic.
    """
    meta = _parse_index_json(folder)
    if meta:
        missing = [fn for fn in meta["files"] if not (folder / fn).exists()]
        return (len(missing) == 0, len(meta["files"]), missing)

    # No index: heuristic — count shard files
    shards = list(folder.glob("model-*-of-*.safetensors"))
    if shards:
        return (True, len(shards), [])
    # Single-file fallback
    single = folder / "model.safetensors"
    if single.exists():
        return (True, 1, [])
    return (False, 0, [])


def _score_snapshot(p: Path) -> tuple:
    """Higher is better: (valid_flag, shard_count, mtime)."""
    valid, count, _ = _snapshot_is_valid(p)
    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = 0
    return (1 if valid else 0, count, mtime)


def find_best_local_snapshot(repo_id: str) -> str | None:
    best = (0, -1, -1)  # valid, count, mtime
    best_path = None
    for root in _roots():
        for pat in _snapshot_globs(repo_id):
            for p in Path(root).glob(pat):
                score = _score_snapshot(p)
                if score > best:
                    best = score
                    best_path = str(p)
    return best_path


def report_cache(repo_id: str, resolved_path: str | None):
    print("\n===== HF cache resolver (validated) =====")
    print("HF_HOME           :", os.environ.get("HF_HOME"))
    print("HF_HUB_CACHE      :", os.environ.get("HF_HUB_CACHE"))
    print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
    print("repo_id           :", repo_id)
    print("Chosen snapshot   :", resolved_path or "<none>")
    print("Roots searched    :", ", ".join(_roots()))
    print("=========================================\n")


# ------------------------- Task utils -------------------------
def normalize_text(text: str) -> str:
    if not text: return ""
    try:
        text = text.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    t = text.lower().strip()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.replace("–", "-").replace("—", "-")


def extract_final_answer(response: str) -> str:
    """
    Preferred: <final>ANSWER</final>
    Fallbacks:
      - Unsloth chat-template block: <|channel|>final<|message|> ... <|end|>
      - 'assistantfinal' sentinel + first line
    Returns a single trimmed line.
    """
    if not response:
        return ""

    text = response

    # 1) <final> ... </final>
    m = re.search(r"<\s*final\s*>(.*?)</\s*final\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        chunk = m.group(1).strip()
        line = next((ln for ln in chunk.splitlines() if ln.strip()), "")
        return line.strip(" :\t-—")

    # 2) Unsloth-style channel block
    m = re.search(
        r"(?:<\|channel\|\>\s*final\s*<\|message\|\>)(.*?)(?:<\|end\|\>|<\|eot_id\|\>|<\|endoftext\|\>|<\|channel\|\>|</s>|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        chunk = m.group(1)
        line = next((ln for ln in chunk.splitlines() if ln.strip()), "")
        line = re.sub(r"</s>|<\|end\|>|<\|eot_id\|>|<\|endoftext\|>", "", line, flags=re.IGNORECASE)
        return line.strip(" :\t-—")

    # 3) Legacy 'assistantfinal'
    low = text.lower()
    key = "assistantfinal"
    if key in low:
        tail = text[low.rfind(key) + len(key):]
        diag = tail.splitlines()[0]
        diag = re.sub(r"</s>|<\|end\|>|<\|eot_id\|>|<\|endoftext\|>", "", diag, flags=re.IGNORECASE)
        return diag.strip(" :\t-—")

    return ""


def create_detailed_prompt(case) -> str:
    """
    The model is asked to produce exactly two blocks:
      1) <analysis> ... </analysis>  -> free-form reasoning
      2) <final>DIAGNOSIS</final>    -> must be exactly one option copied verbatim from the list below
    """
    dd_formatted = "\n".join(case["differential_diagnosis"])
    return f"""You are an expert radiologist demonstrating structured diagnostic reasoning.
Respond in EXACTLY two XML-style blocks:

<analysis>
Write your systematic reasoning for this case in concise paragraphs or bullets:
1) Connect symptoms to imaging findings.
2) Map each differential: explain supporting and contradicting evidence.
3) Systematically eliminate less likely options.
4) Converge to the single most likely diagnosis.
Keep this section readable and evidence-based.
</analysis>

<final>
IMPORTANT: Output ONLY the final diagnosis as a SINGLE LINE, copied VERBATIM from the Differential diagnoses list below.
- Choose exactly ONE option from the list, character-for-character (same spelling, capitalization, punctuation).
- Do NOT add any extra words, qualifiers, numbering, or formatting.
</final>

Case presentation:

{case["combined_description"]}

Differential diagnoses to consider (pick ONE verbatim for <final>):
{dd_formatted}
"""


def prepare_test_data(df: pd.DataFrame):
    out = []
    for _, row in df.iterrows():
        out.append({
            "case_id": row["case_id"],
            "combined_description": str(row["PostDescription"]),
            "differential_diagnosis": [d.strip() for d in str(row["DifferentialDiagnosisList"]).split(",")],
            "diagnosis": str(row["FinalDiagnosis"]),
        })
    return out


def apply_chat_template_or_fallback(tokenizer, user_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"User:\n{user_text}\n\nAssistant:"


def append_row_to_csv(out_csv: str, row: dict, header_order: list[str] | None):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(out_csv)
    if header_order:
        cols = [c for c in header_order if c in df.columns] + [c for c in df.columns if c not in header_order]
        df = df[cols]
    df.to_csv(out_csv, index=False, mode="a", header=write_header, encoding="utf-8")


def load_processed_ids_if_resuming(out_csv: str) -> set:
    if not os.path.exists(out_csv): return set()
    try:
        done = pd.read_csv(out_csv, usecols=["case_id"])["case_id"].astype(str).tolist()
    except Exception:
        done = pd.read_csv(out_csv)["case_id"].astype(str).tolist()
    return set(done)


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Eurorad evaluation (Unsloth), validated local snapshot resolver.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b", help="Repo id or local directory.")
    ap.add_argument("--snapshot_path", default=os.environ.get("PREFER_SNAPSHOT_DIR", ""), help="Explicit snapshot dir (overrides resolver).")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--local_only", action="store_true", help="Do not contact network; error if no valid local snapshot.")
    ap.add_argument("--num_beams", type=int, default=13)
    ap.add_argument("--num_beam_groups", type=int, default=13)
    ap.add_argument("--diversity_penalty", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=3000)
    args = ap.parse_args()

    # Decide load target
    model_arg = args.model
    chosen_path = None

    if args.snapshot_path:
        if not os.path.isdir(args.snapshot_path):
            raise FileNotFoundError(f"--snapshot_path not found: {args.snapshot_path}")
        # Validate explicitly provided path
        valid, count, miss = _snapshot_is_valid(Path(args.snapshot_path))
        if not valid:
            raise FileNotFoundError(f"--snapshot_path invalid (missing shards): {args.snapshot_path}\nMissing: {miss[:5]}{' ...' if len(miss)>5 else ''}")
        chosen_path = args.snapshot_path
    elif os.path.isdir(model_arg):
        # Direct local directory
        valid, count, miss = _snapshot_is_valid(Path(model_arg))
        if not valid:
            raise FileNotFoundError(f"--model points to a folder without shards: {model_arg}\nMissing: {miss[:5]}{' ...' if len(miss)>5 else ''}")
        chosen_path = model_arg
    else:
        # Resolve a valid snapshot under caches
        best = find_best_local_snapshot(model_arg)
        report_cache(model_arg, best)
        if best is None:
            if args.local_only:
                raise FileNotFoundError("No valid local snapshot found and --local_only is set.")
            # fall back to repo id (may download)
            chosen_path = model_arg
        else:
            chosen_path = best

    # Load via Unsloth
    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=chosen_path,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,          # no quantization
        fast_inference=False,        # stay close to eager behavior
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    model.eval()
    print(f"Loaded from: {chosen_path}")
    print("Device:", next(model.parameters()).device, "| Max seq len:", args.max_seq_len)

    # Data
    df = pd.read_csv(args.input_csv)
    required = {"case_id", "PostDescription", "DifferentialDiagnosisList", "FinalDiagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")
    cases = prepare_test_data(df)
    print(f"Prepared {len(cases)} cases")

    # Resume
    done_ids = load_processed_ids_if_resuming(args.output_csv) if args.resume else set()
    if done_ids:
        print(f"Resume: {len(done_ids)} rows already in {args.output_csv}")

    header_order = [
        "case_id", "actual_sample_number", "prompt_type", "decoding_method",
        "num_beams", "num_beam_groups", "diversity_penalty", "ground_truth",
        "available_options", "user_prompt", "input_token_count",
        "final_chosen_answer", "final_chosen_votes", "final_chosen_from_beam",
        "correct", "all_beam_extracted_answers",
    ] + sum(([f"beam_{i}_text", f"beam_{i}_final"] for i in range(1, args.num_beams + 1)), [])

    acc = []

    for idx, case in enumerate(cases[args.start_index:], start=args.start_index):
        case_id = str(case["case_id"])
        if args.resume and case_id in done_ids:
            print(f"[{idx+1}/{len(cases)}] Skip (already processed): case_id={case_id}")
            continue

        gt = str(case["diagnosis"]).strip()
        user_prompt = create_detailed_prompt(case)
        formatted = apply_chat_template_or_fallback(tokenizer, user_prompt)

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
        inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}
        input_token_count = int(inputs["input_ids"].shape[1])

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            num_return_sequences=args.num_beams,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
            trust_remote_code=True,
        )

        print(f"[{idx+1}/{len(cases)}] case_id={case_id} | beams={args.num_beams} groups={args.num_beam_groups} div={args.diversity_penalty}")
        with torch.no_grad():
            sequences = model.generate(**inputs, **gen_kwargs)

        prompt_len = input_token_count
        beam_texts, beam_finals = [], []
        for i in range(args.num_beams):
            gen_tokens = sequences[i][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
            beam_texts.append(gen_text.strip())
            beam_finals.append(extract_final_answer(gen_text))

        # Majority vote
        norm = [normalize_text(x) for x in beam_finals]
        counts = Counter(norm)
        if counts:
            max_count = max(counts.values())
            tie_norms = {a for a, c in counts.items() if c == max_count}
            chosen_idx = next((i for i, a in enumerate(norm) if a in tie_norms), 0)
            chosen_answer, chosen_votes = beam_finals[chosen_idx], int(max_count)
        else:
            chosen_idx, chosen_answer, chosen_votes = 0, (beam_finals[0] if beam_finals else ""), 0

        chosen_from_beam = chosen_idx + 1
        is_correct = int(normalize_text(chosen_answer) == normalize_text(gt))
        acc.append(is_correct)

        row = {
            "case_id": case_id,
            "actual_sample_number": idx + 1,
            "prompt_type": "DETAILED",
            "decoding_method": "DIVERSE_BEAM_MAJORITY",
            "num_beams": args.num_beams,
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty,
            "ground_truth": gt,
            "available_options": " | ".join(case["differential_diagnosis"]),
            "user_prompt": user_prompt,
            "input_token_count": input_token_count,
            "final_chosen_answer": chosen_answer,
            "final_chosen_votes": chosen_votes,
            "final_chosen_from_beam": chosen_from_beam,
            "correct": is_correct,
            "all_beam_extracted_answers": json.dumps(beam_finals, ensure_ascii=False),
        }
        for i in range(args.num_beams):
            row[f"beam_{i+1}_text"] = beam_texts[i]
            row[f"beam_{i+1}_final"] = beam_finals[i]

        append_row_to_csv(args.output_csv, row, header_order)

        mean_acc = np.mean(acc)
        std_acc = np.std(acc) if len(acc) > 1 else 0.0
        print(f"   → chosen (beam {chosen_from_beam}, votes={chosen_votes}): {chosen_answer or '[EMPTY]'} | correct={bool(is_correct)} | running acc={mean_acc:.3f} ± {std_acc:.3f}")

    print("\nDone.")
    if acc:
        print(f"Final Majority Accuracy: {np.mean(acc):.3f} ± {np.std(acc) if len(acc)>1 else 0.0:.3f}")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
