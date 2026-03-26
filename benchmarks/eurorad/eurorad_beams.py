#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"                 
os.environ["HF_PYTORCH_ATTENTION_BACKEND"] = "eager" #force eager attention

from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

from unsloth import FastLanguageModel
from peft import PeftModel

BASE = "openai/gpt-oss-20b"
LORA_PATH = "/home/alhusain/scratch/ondevice-llm/eurorad_MoE_Lprompt/"
MAX_SEQ_LEN = 4096

# Load base model in 4-bit
model_base, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    fast_inference=False,         
    attn_implementation="eager",   
    float8_kv_cache=True,
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(model_base, LORA_PATH)
model.eval()

# Tokenizer tweaks for generation
tokenizer.padding_side = "left"
tokenizer.pad_token    = tokenizer.eos_token

print("Ready. PEFT attached:", isinstance(model, PeftModel))
print("Active adapters:", list(getattr(model, "peft_config", {}).keys()))
print("Device:", next(model.parameters()).device, "| Max seq len:", MAX_SEQ_LEN)


# In[2]:


#Cell 1: Data Loading
import pandas as pd

# Load test data
print("Loading test data...")
test_df = pd.read_csv('/home/alhusain/scratch/ondevice-llm/eurorad_val.csv')
test_df = test_df.iloc[0:].copy()
print(f"Loaded {len(test_df)} test cases")

def prepare_test_data(df):
    """Convert test data format to evaluation format"""
    processed_data = []

    for _, row in df.iterrows():
        # Use combined description as-is
        full_description = row['PostDescription']

        # Process differential diagnosis list
        dd_list = [dd.strip() for dd in str(row['DifferentialDiagnosisList']).split(',')]

        processed_data.append({
            'case_id': row['case_id'],
            'combined_description': full_description,
            'differential_diagnosis': dd_list,
            'diagnosis': row['FinalDiagnosis']
        })

    return processed_data

# Prepare test data
test_data = prepare_test_data(test_df)
print(f"Test data prepared: {len(test_data)} cases")
print(f"Sample case ID: {test_data[0]['case_id']}")
print(f"Sample diagnosis: {test_data[0]['diagnosis']}")


# In[3]:


import os
import torch
import pandas as pd
import re
import numpy as np
from datetime import datetime
import json
from collections import Counter

def normalize_text(text):

    if not text:
        return ""

    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)  # Keep only alphanumeric, spaces, hyphens
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    text = text.replace('–', '-').replace('—', '-')  # Normalize dash types
    text = text.replace('"', '').replace('"', '').replace('"', '')  # Remove quotes

    return text

def extract_after_assistantfinal(response):


    if "assistantfinal" not in response.lower():
        return ""

    parts = response.lower().split("assistantfinal")
    if len(parts) < 2:
        return ""

    # Get the final section (everything after assistantfinal)
    final_section = parts[-1].strip()

    # Take only the first line (the diagnosis itself)
    diagnosis = final_section.split('\n')[0].strip()

    # Clean up any remaining special tokens
    diagnosis = diagnosis.replace('</s>', '').replace('<|end|>', '').replace('<|eot_id|>', '').strip()

    return diagnosis

def create_detailed_prompt(case_data):
    combined_description = case_data['combined_description']
    dd_list = case_data['differential_diagnosis']
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


def run_evaluation_diverse_beam_majority_noscores(
    model, tokenizer, test_data, start_index=0, model_name="gptoss20b_Oct7_finetuned"
):
    # Output folder
    today = datetime.now().strftime("%Y%m%d")
    output_folder = f"{today}_{model_name}_DETAILED_DIVERSEBEAM_MAJ_eval_b9_d0p5_noScores"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Cases
    remaining_cases = test_data[start_index:]
    n_cases = len(remaining_cases)
    all_results = []
    case_accuracies = []

    # Diverse beam params
    NUM_BEAMS = 13
    NUM_BEAM_GROUPS = 13      # 1 group per beam for maximum diversity
    DIVERSITY_PENALTY = 0.5
    NUM_RETURN = 13      
    MAX_NEW_TOKENS = 3000

    print(f"=== EVALUATION: DETAILED PROMPT (DIVERSE BEAM SEARCH, MAJORITY VOTE, NO SCORES) ===")
    print(f"Total cases: {n_cases} (samples {start_index+1} to {len(test_data)})")
    print(f"Decoding: num_beams={NUM_BEAMS}, num_beam_groups={NUM_BEAM_GROUPS}, diversity_penalty={DIVERSITY_PENALTY}")
    print("Final answer = most frequent beam answer; tie -> earliest beam")
    print("=" * 80)

    for case_idx, case in enumerate(remaining_cases):
        actual_sample_num = start_index + case_idx
        case_id = case['case_id']
        ground_truth = str(case['diagnosis']).strip()

        print(f"\n{'='*80}")
        print(f"CASE ID: {case_id} (Sample {actual_sample_num+1}/{len(test_data)})")
        print(f"Progress: {((actual_sample_num+1)/len(test_data))*100:.1f}%")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*80}")

        # Build prompt
        user_prompt = create_detailed_prompt(case)
        print("PROMPT FORMAT: DETAILED (with Chain-of-Thought instructions)")
        print("-" * 40)
        preview = user_prompt[:300] + "..." if len(user_prompt) > 300 else user_prompt
        print(preview)
        print("-" * 40)

        # Chat template 
        messages = [{"role": "user", "content": user_prompt}]
        formatted_input = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Tokenize
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        # Move to same device as model (no dtype casting)
        inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}
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
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                beam_texts.append(gen_text)
                beam_final_answers.append(extract_after_assistantfinal(gen_text))

            # Majority vote (normalize for counting, tie -> earliest)
            norm_answers = [normalize_text(a) for a in beam_final_answers]
            counts = Counter(norm_answers)

            if len(counts) == 0:
                chosen_beam_idx = 0
                chosen_final_answer = beam_final_answers[0] if beam_final_answers else ""
                chosen_votes = 0
            else:
                max_count = max(counts.values())
                # set of normalized answers that tie for max
                tie_norms = {ans for ans, c in counts.items() if c == max_count}
                # earliest beam whose normalized answer is in tie set
                chosen_beam_idx = next((i for i, a in enumerate(norm_answers) if a in tie_norms), 0)
                chosen_final_answer = beam_final_answers[chosen_beam_idx]
                chosen_votes = int(max_count)

            chosen_from_beam = chosen_beam_idx + 1

            # Evaluate correctness vs GT using normalized comparison
            is_correct = normalize_text(chosen_final_answer) == normalize_text(ground_truth)
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
                "available_options": " | ".join(case['differential_diagnosis']),
                "user_prompt": user_prompt,
                "input_token_count": input_token_count,
                # Majority-vote decision
                "final_chosen_answer": chosen_final_answer,
                "final_chosen_votes": chosen_votes,
                "final_chosen_from_beam": int(chosen_from_beam),
                "correct": 1 if is_correct else 0,
                # Aggregate list for analysis
                "all_beam_extracted_answers": json.dumps(beam_final_answers, ensure_ascii=False),
            }

            # Add per-beam columns
            for i in range(NUM_RETURN):
                row[f"beam_{i+1}_text"] = beam_texts[i]
                row[f"beam_{i+1}_final"] = beam_final_answers[i]

            all_results.append(row)

            print(f"Chosen answer (beam {chosen_from_beam}, votes={chosen_votes}): {chosen_final_answer or '[EMPTY]'}")
            print(f"Correct? {'YES' if is_correct else 'NO'}")
            print(f"Running Mean Accuracy (majority-vote): {np.mean(case_accuracies):.3f}")

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
                "available_options": " | ".join(case['differential_diagnosis']),
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
                row[f"beam_{i+1}_text"] = ""
                row[f"beam_{i+1}_final"] = ""
            all_results.append(row)

        # CHECKPOINT every 5 cases or at end
        if ((case_idx + 1) % 5 == 0) or ((case_idx + 1) == n_cases):
            results_df = pd.DataFrame(all_results)
            timestamp = datetime.now().strftime("%H%M%S")
            ckpt_name = f"DIVERSEBEAM_MAJ_noScores_checkpoint_sample_{actual_sample_num+1}_{timestamp}.csv"
            ckpt_path = os.path.join(output_folder, ckpt_name)
            results_df.to_csv(ckpt_path, index=False, encoding="utf-8")
            print(f"\n*** CHECKPOINT SAVED: {ckpt_path} ***")
            mean_acc = np.mean(case_accuracies) if case_accuracies else 0.0
            std_acc = np.std(case_accuracies) if len(case_accuracies) > 1 else 0.0
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
                "accuracy_display_majority": f"{mean_acc:.3f} ± {std_acc:.3f}",
            }
            sum_name = f"DIVERSEBEAM_MAJ_noScores_summary_sample_{actual_sample_num+1}_{timestamp}.txt"
            sum_path = os.path.join(output_folder, sum_name)
            with open(sum_path, "w") as f:
                f.write("=== DETAILED PROMPT EVALUATION SUMMARY (DIVERSE BEAM + MAJORITY, NO SCORES) ===\n\n")
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")
            print(f"*** Current Majority Accuracy: {mean_acc:.3f} ± {std_acc:.3f} ***\n")

    # Final save
    results_df = pd.DataFrame(all_results)
    final_mean = np.mean(case_accuracies) if case_accuracies else 0.0
    final_std = np.std(case_accuracies) if len(case_accuracies) > 1 else 0.0
    timestamp = datetime.now().strftime("%H%M%S")
    final_name = f"FINAL_DIVERSEBEAM_MAJ_noScores_{model_name}_{timestamp}.csv"
    final_path = os.path.join(output_folder, final_name)
    results_df.to_csv(final_path, index=False, encoding="utf-8")

    print("\n" + "#" * 80)
    print("DETAILED PROMPT EVALUATION COMPLETE (DIVERSE BEAM + MAJORITY VOTE, NO SCORES)")
    print("#" * 80)
    print(f"Total cases: {n_cases}")
    print(f"Final Majority Accuracy: {final_mean:.3f} ± {final_std:.3f}")
    print(f"Final CSV: {final_path}")
    print(f"Output folder: {output_folder}")
    print("#" * 80)

    return results_df, (final_mean, final_std), case_accuracies


results_diverse_beam_maj_ns, (mean_diverse_maj_ns, std_diverse_maj_ns), acc_diverse_maj_ns = run_evaluation_diverse_beam_majority_noscores(
    model, tokenizer, test_data, start_index=0, model_name="gptoss20b_Oct7_finetuned"
)

