import os

# Must be set BEFORE importing unsloth.
# Unsloth's default MoE Triton grouped-GEMM backend requires triton>=3.2
# (triton.language.make_tensor_descriptor), which conflicts with torch 2.5.1's
# pinned triton==3.1.0. Switching to the torch backend avoids the crash.
os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduces fragmentation-driven OOM

import pandas as pd
from datasets import Dataset
from unsloth import FastModel  # FastModel is required for MoE architectures (35B-A3B)
from trl import SFTConfig, SFTTrainer

# ==========================================
# MULTI-GPU NOTE:
# Run with 2 GPUs: CUDA_VISIBLE_DEVICES=0,1 python qwen35b_finetune.py
# Run with all 4:  python qwen35b_finetune.py  (device_map="balanced" auto-distributes)
# ==========================================

# ==========================================
# 1. MODEL & TOKENIZER SETUP
# ==========================================
MODEL_NAME = "unsloth/Qwen3.5-35B-A3B"
MAX_SEQ_LEN = 2048  # Halved from 4096 — GatedDeltaNet memory scales quadratically with seq_len

print(f"Loading model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_SEQ_LEN}")

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,      # MoE QLoRA is NOT recommended per Unsloth guide
    load_in_16bit=True,      # bf16 LoRA — the correct setup for 35B-A3B
    full_finetuning=False,
    device_map="balanced",   # Distribute model layers evenly across available GPUs
)

# For MoE models, explicitly list target modules (avoids accidentally targeting
# router/gating layers, which Unsloth disables by default for stability)
model = FastModel.get_peft_model(
    model,
    r=16,             # Sufficient for 35B — larger r on a big model risks overfitting
    lora_alpha=32,    # 2x r is a stable default
    lora_dropout=0,      # Must be 0 for MoE models — ParamWrapper (used for expert layers) does not support dropout
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's own checkpointing, lower VRAM
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("Configuring tokenizer...")
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
model.print_trainable_parameters()

# ==========================================
# 2. DATASET PROCESSING
# ==========================================
print("Loading enhanced CSV with Qwen122B reasoning...")
enhanced_df = pd.read_csv('Enhanced_Training_Data_with_Qwen122B_Reasoning.csv')

matched_df = enhanced_df[enhanced_df['reasoning_matched'] == True].copy()
print(f"Processing {len(matched_df)} cases with Qwen122B reasoning")


def enhanced_formatting_prompts_func(examples):
    """Format with GPToss120B reasoning"""
    convos = []

    for i in range(len(examples["PostDescription"])):
        combined_description = examples["PostDescription"][i]
        differential_diagnosis = examples["DifferentialDiagnosisList"][i]
        diagnosis = examples["FinalDiagnosis"][i]
        reasoning = examples["gptoss120b_reasoning"][i]

        dd_list = [dd.strip() for dd in str(differential_diagnosis).split(',')]
        dd_formatted = "\n".join(dd_list)

        user_prompt = f"""You are an expert radiologist demonstrating step-by-step diagnostic reasoning.

Case presentation:

{combined_description}

Differential diagnoses to consider:
{dd_formatted}

Generate systematic Chain-of-Thought reasoning that shows how clinicians think through cases:

1. **Connect symptoms to findings**: Link clinical presentation with imaging observations
2. **Map to differentials**: Show how findings support or contradict each differential diagnosis
3. **Systematic elimination**: Explicitly rule out less likely options with reasoning
4. **Converge to answer**: Demonstrate the logical path to the correct diagnosis"""

        # Manually embed <think> tags inside assistant content for Qwen3.5 thinking style
        assistant_response = f"<think>\n{reasoning}\n</think>\n\n{diagnosis}"

        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
        convos.append(conversation)

    return {"messages": convos}


print("\nConverting to HuggingFace Dataset...")
dataset = Dataset.from_pandas(matched_df)

print("Applying enhanced formatting...")
dataset = dataset.map(enhanced_formatting_prompts_func, batched=True)


def final_formatting_func(examples):
    """Apply standard chat template"""
    convos = examples["messages"]
    texts = []
    for convo in convos:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_token=False
        )
        texts.append(text)
    return {"text": texts}


print("Applying chat template...")
dataset = dataset.map(final_formatting_func, batched=True)

print("Sample formatted example:")
print("=" * 50)
print(dataset[0]['text'])

# ==========================================
# 3. WANDB & TRAINING
# ==========================================
os.environ["WANDB_API_KEY"] = "3b3c04cb5ade90dcfc9134cdeeced5565bdbaa20"
os.environ["WANDB_PROJECT"] = "eurorad_qwen3.5_35b"

print("\nStarting Thinking Style Training with Qwen3.5 35B-A3B (MoE)")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,   # Keep at 1; 35B-A3B is memory-heavy
        gradient_accumulation_steps=16,  # Effective batch = 16 per GPU
        num_train_epochs=3,
        learning_rate=5e-5,              # Lower LR for a larger model (was 1e-4 for 9B)
        max_seq_length=MAX_SEQ_LEN,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        report_to="wandb",
        run_name="eurorad_thinking_qwen3.5_35b",
        output_dir="eurorad_thinking_qwen3.5_35b_output",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,              # Reduced from 10; 35B checkpoints are large
        gradient_checkpointing=False,    # Handled by use_gradient_checkpointing="unsloth" above
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=3407,
        data_seed=3407,
        bf16=True,
        fp16=False,
    ),
)

print("Starting thinking style training...")
trainer.train()

# ==========================================
# 4. SAVE / EXPORT
# ==========================================
print("\nSaving LoRA adapters...")
model.save_pretrained("eurorad_thinking_qwen3.5_35b_lora")
tokenizer.save_pretrained("eurorad_thinking_qwen3.5_35b_lora")

# Export to GGUF for single-GPU inference (Q4_K_M fits comfortably on one 48GB GPU)
print("Exporting to GGUF (q4_k_m) for single-GPU inference...")
model.save_pretrained_gguf(
    "eurorad_thinking_qwen3.5_35b_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
print("Done. GGUF model saved to: eurorad_thinking_qwen3.5_35b_gguf")
