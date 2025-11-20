#!/usr/bin/env python
import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import pandas as pd
import numpy as np
from torch.backends.cuda import sdp_kernel
from unsloth import FastLanguageModel
from peft import PeftModel

# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "openai/gpt-oss-20b"
DEFAULT_LORA_PATH = "/home/alhusain/scratch/ondevice-llm/eurorad_MoE_Lprompt/"
DEFAULT_MAX_SEQ_LEN = 4096

DEFAULT_TEST_CSV = "/home/alhusain/scratch/ondevice-llm/eurorad_val.csv"
DEFAULT_MODEL_NAME = "gptoss20b_Oct7_finetuned"

DEFAULT_NUM_BEAMS = 13
DEFAULT_NUM_BEAM_GROUPS = 13
DEFAULT_DIVERSITY_PENALTY = 0.5
DEFAULT_MAX_NEW_TOKENS = 3000

DEFAULT_START_INDEX = 0
DEFAULT_CUDA_VISIBLE_DEVICES = "2"
DEFAULT_ATTENTION_BACKEND = "eager"


# --------------------------------------------------------------------------------------
# Environment / model setup
# --------------------------------------------------------------------------------------
def setup_environment(cuda_visible_devices: str, attention_backend: str) -> None:
    """
    Set environment variables and CUDA attention backend, matching the notebook behavior.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["HF_PYTORCH_ATTENTION_BACKEND"] = attention_backend  # force eager attention

    sdp_kernel(enable_flash=False, enable_mem_efficienct=False, enable_math=True)


def load_model(base_model: str, lora_path: str, max_seq_len: int):
    """
    Load base model in 4-bit and attach LoRA adapter, same as in your notebook.
    """
    # Load base model in 4-bit
    model_base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        fast_inference=False,
        attn_implementation="eager",
        float8_kv_cache=True,
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(model_base, lora_path)
    model.eval()

    # Tokenizer tweaks for generation
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    print("Ready. PEFT attached:", isinstance(model, PeftModel))
    print("Active adapters:", list(getattr(model, "peft_config", {}).keys()))
    print("Device:", next(model.parameters()).device, "| Max seq len:", max_seq_len)

    return model, tokenizer


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------
def prepare_test_data(df: pd.DataFrame):
    """Convert test data format to evaluation format (unchanged)."""
    processed_data = []

    for _, row in df.iterrows():
        # Use combined description as-is
        full_description = row["PostDescription"]

        # Process differential diagnosis list
        dd_list = [dd.strip() for dd in str(row["DifferentialDiagnosisList"]).split(",")]

        processed_data.append(
            {
                "case_id": row["case_id"],
                "combined_description": full_description,
                "differential_diagnosis": dd_list,
                "diagnosis": row["FinalDiagnosis"],
            }
        )

    return processed_data


# --------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)  # Keep only alphanumeric, spaces, hyphens
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    text = text.replace("–", "-").replace("—", "-")  # Normalize dash types
    text = text.replace('"', "").replace('"', "").replace('"', "")  # Remove quotes

    return text


def extract_after_assistantfinal(response: str) -> str:
    if "assistantfinal" not in response.lower():
        return ""

    parts = response.lower().split("assistantfinal")
    if len(parts) < 2:
        return ""

    # Get the final section (everything after assistantfinal)
    final_section = parts[-1].strip()

    # Take only the first line (the diagnosis itself)
    diagnosis = final_section.split("\n")[0].strip()

    # Clean up any remaining special tokens
    diagnosis = (
        diagnosis.replace("</s>", "")
        .replace("<|end|>", "")
        .replace("<|eot_id|>", "")
        .strip()
    )

    return diagnosis


def create_detailed_prompt(case_data: dict) -> str:
    combined_description = case_data["combined_description"]
    dd_list = case_data["differential_diagnosis"]
    dd_formatted = "\n".join(dd_list)

    prompt = f"""You are an expert radiologist demonstrating step-by-step diagnostic reasoning.

Case presentation:

{combined_description}

Differential diagnoses to consider:
{dd_formatted}

Generate systematic Chain-of-Thought reasoning that shows how clinicians think through cases:

1. **Connect symptoms to findings**: Link clinical presentation with imaging observations
2. **Map to differentials**: Show how findings support or contradict each differential diagnosis
3. **Systematic elimination**: Explicitly rule out less likely options with reasoning
4. **Converge to answer**: Demonstrate the logical path to the correct diagnosis"""
    return prompt


# --------------------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------------------
def run_evaluation_diverse_beam_majority_noscores(
    model,
    tokenizer,
    test_data,
    start_index: int = 0,
    model_name: str = DEFAULT_MODEL_NAME,
    num_beam_groups: int = DEFAULT_NUM_BEAM_GROUPS,
    diversity_penalty: float = DEFAULT_DIVERSITY_PENALTY,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    # Output folder
    today = datetime.now().strftime("%Y%m%d")
    output_folder = (
        f"{today}_{model_name}_DETAILED_DIVERSEBEAM_MAJ_eval_b9_d0p5_noScores"
    )
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Cases
    remaining_cases = test_data[start_index:]
    n_cases = len(remaining_cases)
    all_results = []
    case_accuracies = []

    # Diverse beam params
    NUM_BEAMS = DEFAULT_NUM_BEAMS
    NUM_BEAM_GROUPS = num_beam_groups
    DIVERSITY_PENALTY = diversity_penalty
    NUM_RETURN = NUM_BEAMS
    MAX_NEW_TOKENS = max_new_tokens

    print(
        "=== EVALUATION: DETAILED PROMPT (DIVERSE BEAM SEARCH, MAJORITY VOTE, NO SCORES) ==="
    )
    print(
        f"Total cases: {n_cases} (samples {start_index + 1} to {len(test_data)})"
    )
    print(
        f"Decoding: num_beams={NUM_BEAMS}, num_beam_groups={NUM_BEAM_GROUPS}, "
        f"diversity_penalty={DIVERSITY_PENALTY}, max_new_tokens={MAX_NEW_TOKENS}"
    )
    print("Final answer = most frequent beam answer; tie -> earliest beam")
    print("=" * 80)

    for case_idx, case in enumerate(remaining_cases):
        actual_sample_num = start_index + case_idx
        case_id = case["case_id"]
        ground_truth = str(case["diagnosis"]).strip()

        print(f"\n{'=' * 80}")
        print(
            f"CASE ID: {case_id} (Sample {actual_sample_num + 1}/{len(test_data)})"
        )
        print(f"Progress: {((actual_sample_num + 1) / len(test_data)) * 100:.1f}%")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'=' * 80}")

        # Build prompt
        user_prompt = create_detailed_prompt(case)
        print("PROMPT FORMAT: DETAILED (with Chain-of-Thought instructions)")
        print("-" * 40)
        preview = (
            user_prompt[:300] + "..." if len(user_prompt) > 300 else user_prompt
        )
        print(preview)
        print("-" * 40)

        # Chat template
        messages = [{"role": "user", "content": user_prompt}]
        formatted_input = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Tokenize
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_SEQ_LEN,
        )
        # Move to same device as model (no dtype casting)
        inputs = {
            k: v.to(model.device, non_blocking=True) for k, v in inputs.items()
        }
        input_token_count = inputs["input_ids"].shape[1]
        print(f"Input tokens: {input_token_count}")

        try:
            with torch.no_grad():
                # Diverse beam search (no sampling, no scores dict)
                sequences = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=NUM_BEAMS,
                    num_beam_groups=NUM_BEAM_GROUPS,
                    diversity_penalty=DIVERSITY_PENALTY,
                    num_return_sequences=NUM_RETURN,
                    early_stopping=True,
                    return_dict_in_generate=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode each beam (excluding prompt tokens)
            beam_texts = []
            beam_final_answers = []
            for i in range(NUM_RETURN):
                gen_tokens = sequences[i][input_token_count:]
                gen_text = tokenizer.decode(
                    gen_tokens, skip_special_tokens=True
                ).strip()
                beam_texts.append(gen_text)
                beam_final_answers.append(
                    extract_after_assistantfinal(gen_text)
                )

            # Majority vote (normalize for counting, tie -> earliest)
            norm_answers = [normalize_text(a) for a in beam_final_answers]
            counts = Counter(norm_answers)

            if len(counts) == 0:
                chosen_beam_idx = 0
                chosen_final_answer = (
                    beam_final_answers[0] if beam_final_answers else ""
                )
                chosen_votes = 0
            else:
                max_count = max(counts.values())
                # set of normalized answers that tie for max
                tie_norms = {ans for ans, c in counts.items() if c == max_count}
                # earliest beam whose normalized answer is in tie set
                chosen_beam_idx = next(
                    (i for i, a in enumerate(norm_answers) if a in tie_norms), 0
                )
                chosen_final_answer = beam_final_answers[chosen_beam_idx]
                chosen_votes = int(max_count)

            chosen_from_beam = chosen_beam_idx + 1

            # Evaluate correctness vs GT using normalized comparison
            is_correct = normalize_text(chosen_final_answer) == normalize_text(
                ground_truth
            )
            case_accuracies.append(1.0 if is_correct else 0.0)

            # Build row with all beams + majority-vote result
            row = {
                "case_id": case_id,
                "actual_sample_number": actual_sample_num + 1,
                "prompt_type": "DETAILED",
                "decoding_method": "DIVERSE_BEAM_MAJORITY",
                "num_beams": NUM_BEAMS,
                "num_beam_groups": NUM_BEAM_GROUPS,
                "diversity_penalty": DIVERSITY_PENALTY,
                "ground_truth": ground_truth,
                "available_options": " | ".join(
                    case["differential_diagnosis"]
                ),
                "user_prompt": user_prompt,
                "input_token_count": input_token_count,
                # Majority-vote decision
                "final_chosen_answer": chosen_final_answer,
                "final_chosen_votes": chosen_votes,
                "final_chosen_from_beam": int(chosen_from_beam),
                "correct": 1 if is_correct else 0,
                # Aggregate list for analysis
                "all_beam_extracted_answers": json.dumps(
                    beam_final_answers, ensure_ascii=False
                ),
            }

            # Add per-beam columns
            for i in range(NUM_RETURN):
                row[f"beam_{i + 1}_text"] = beam_texts[i]
                row[f"beam_{i + 1}_final"] = beam_final_answers[i]

            all_results.append(row)

            print(
                f"Chosen answer (beam {chosen_from_beam}, votes={chosen_votes}): "
                f"{chosen_final_answer or '[EMPTY]'}"
            )
            print(f"Correct? {'YES' if is_correct else 'NO'}")
            print(
                f"Running Mean Accuracy (majority-vote): "
                f"{np.mean(case_accuracies):.3f}"
            )

        except Exception as e:
            print(f"ERROR in diverse beam generation: {str(e)}")
            # Record a row with placeholders to keep schema stable
            row = {
                "case_id": case_id,
                "actual_sample_number": actual_sample_num + 1,
                "prompt_type": "DETAILED",
                "decoding_method": "DIVERSE_BEAM_MAJORITY",
                "num_beams": NUM_BEAMS,
                "num_beam_groups": NUM_BEAM_GROUPS,
                "diversity_penalty": DIVERSITY_PENALTY,
                "ground_truth": ground_truth,
                "available_options": " | ".join(
                    case["differential_diagnosis"]
                ),
                "user_prompt": user_prompt,
                "input_token_count": input_token_count,
                "final_chosen_answer": "",
                "final_chosen_votes": 0,
                "final_chosen_from_beam": 1,
                "correct": 0,
                "all_beam_extracted_answers": json.dumps([]),
                "error": str(e),
            }
            for i in range(NUM_RETURN):
                row[f"beam_{i + 1}_text"] = ""
                row[f"beam_{i + 1}_final"] = ""
            all_results.append(row)

        # CHECKPOINT every 5 cases or at end
        if ((case_idx + 1) % 5 == 0) or ((case_idx + 1) == n_cases):
            results_df = pd.DataFrame(all_results)
            timestamp = datetime.now().strftime("%H%M%S")
            ckpt_name = (
                f"DIVERSEBEAM_MAJ_noScores_checkpoint_sample_"
                f"{actual_sample_num + 1}_{timestamp}.csv"
            )
            ckpt_path = os.path.join(output_folder, ckpt_name)
            results_df.to_csv(ckpt_path, index=False, encoding="utf-8")
            print(f"\n*** CHECKPOINT SAVED: {ckpt_path} ***")
            mean_acc = np.mean(case_accuracies) if case_accuracies else 0.0
            std_acc = (
                np.std(case_accuracies) if len(case_accuracies) > 1 else 0.0
            )
            summary = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": model_name,
                "prompt_type": "DETAILED",
                "decoding_method": "DIVERSE_BEAM_MAJORITY",
                "num_beams": NUM_BEAMS,
                "num_beam_groups": NUM_BEAM_GROUPS,
                "diversity_penalty": DIVERSITY_PENALTY,
                "current_sample": actual_sample_num + 1,
                "total_samples": len(test_data),
                "cases_completed": case_idx + 1,
                "mean_accuracy_majority": f"{mean_acc:.4f}",
                "std_accuracy_majority": f"{std_acc:.4f}",
                "accuracy_display_majority": (
                    f"{mean_acc:.3f} ± {std_acc:.3f}"
                ),
            }
            sum_name = (
                f"DIVERSEBEAM_MAJ_noScores_summary_sample_"
                f"{actual_sample_num + 1}_{timestamp}.txt"
            )
            sum_path = os.path.join(output_folder, sum_name)
            with open(sum_path, "w") as f:
                f.write(
                    "=== DETAILED PROMPT EVALUATION SUMMARY "
                    "(DIVERSE BEAM + MAJORITY, NO SCORES) ===\n\n"
                )
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")
            print(
                f"*** Current Majority Accuracy: {mean_acc:.3f} ± {std_acc:.3f} ***\n"
            )

    # Final save
    results_df = pd.DataFrame(all_results)
    final_mean = np.mean(case_accuracies) if case_accuracies else 0.0
    final_std = np.std(case_accuracies) if len(case_accuracies) > 1 else 0.0
    timestamp = datetime.now().strftime("%H%M%S")
    final_name = (
        f"FINAL_DIVERSEBEAM_MAJ_noScores_{model_name}_{timestamp}.csv"
    )
    final_path = os.path.join(output_folder, final_name)
    results_df.to_csv(final_path, index=False, encoding="utf-8")

    print("\n" + "#" * 80)
    print(
        "DETAILED PROMPT EVALUATION COMPLETE (DIVERSE BEAM + MAJORITY VOTE, NO SCORES)"
    )
    print("#" * 80)
    print(f"Total cases: {n_cases}")
    print(f"Final Majority Accuracy: {final_mean:.3f} ± {final_std:.3f}")
    print(f"Final CSV: {final_path}")
    print(f"Output folder: {output_folder}")
    print("#" * 80)

    return results_df, (final_mean, final_std), case_accuracies


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Eurorad evaluation with GPT-OSS-20B + LoRA using diverse beam search and majority vote."
        )
    )
    ap.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model name for Unsloth FastLanguageModel.",
    )
    ap.add_argument(
        "--lora-path",
        default=DEFAULT_LORA_PATH,
        help="Path to LoRA adapter directory.",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Maximum sequence length.",
    )
    ap.add_argument(
        "--test-csv",
        type=Path,
        default=Path(DEFAULT_TEST_CSV),
        help="Path to test CSV file.",
    )
    ap.add_argument(
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help="Start index into test dataset.",
    )
    ap.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name tag used in output filenames.",
    )
    ap.add_argument(
        "--num-beam-groups",
        type=int,
        default=DEFAULT_NUM_BEAM_GROUPS,
        help="Number of beam groups for diverse beam search.",
    )
    ap.add_argument(
        "--diversity-penalty",
        type=float,
        default=DEFAULT_DIVERSITY_PENALTY,
        help="Diversity penalty for diverse beam search.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens to generate.",
    )
    ap.add_argument(
        "--cuda-visible-devices",
        default=DEFAULT_CUDA_VISIBLE_DEVICES,
        help="CUDA_VISIBLE_DEVICES value (default: 2).",
    )
    ap.add_argument(
        "--attention-backend",
        default=DEFAULT_ATTENTION_BACKEND,
        help="HF_PYTORCH_ATTENTION_BACKEND value (default: eager).",
    )

    args = ap.parse_args()

    # Environment & model
    setup_environment(args.cuda_visible_devices, args.attention_backend)
    model, tokenizer = load_model(
        args.base_model, args.lora_path, args.max_seq_len
    )

    # Data loading
    print("Loading test data...")
    test_df = pd.read_csv(args.test_csv)
    test_df = test_df.iloc[0:].copy()
    print(f"Loaded {len(test_df)} test cases")

    test_data = prepare_test_data(test_df)
    print(f"Test data prepared: {len(test_data)} cases")
    if len(test_data) > 0:
        print(f"Sample case ID: {test_data[0]['case_id']}")
        print(f"Sample diagnosis: {test_data[0]['diagnosis']}")

    # Run evaluation
    run_evaluation_diverse_beam_majority_noscores(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        start_index=args.start_index,
        model_name=args.model_name,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
