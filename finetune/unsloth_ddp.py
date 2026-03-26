#!/usr/bin/env python3
import os, torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTTrainer, SFTConfig
from transformers.utils import is_flash_attn_2_available

os.environ["UNSLOTH_DISABLE_RL_PATCH"] = "1"   # skip RL trainer patching
os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"  # avoid unsloth_zoo live changes

# 1) Model (LoRA + 4bit stays for VRAM efficiency)
max_seq_length = 2048  # raise if VRAM allows
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-120b",
    dtype=None,                 # auto -> bf16 on A100
    max_seq_length=max_seq_length,
    load_in_4bit=True,          # keeps memory low; increase batch instead of disabling
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 2) Speed knobs
model.config.use_cache = False
if is_flash_attn_2_available():
    model.config.attn_implementation = "flash_attention_2"
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matmuls where relevant

# 3) Dataset → text + packing-friendly
def fmt(examples):
    return {"text": [tokenizer.apply_chat_template(m, tokenize=False,
               reasoning_effort="medium", add_generation_prompt=False)
               for m in examples["messages"]]}

ds = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
ds = standardize_sharegpt(ds)
ds = ds.map(fmt, batched=True, num_proc=8, remove_columns=ds.column_names)

# 4) Trainer (DDP enabled via launch command; see below)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="text",
    packing=True,  # packs many short samples → better GPU fill
    args=SFTConfig(
        bf16=True,
        ddp_backend="nccl",
        per_device_train_batch_size=8,     # increase until ~90% VRAM
        gradient_accumulation_steps=2,     # tune with batch size
        dataloader_num_workers=8,          # try 12–16 if CPU allows
        dataloader_pin_memory=True,
        group_by_length=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,

        learning_rate=2e-4,
        weight_decay=0.01,
        optim="adamw_torch_fused",         # faster than 8-bit on A100 for LoRA
        lr_scheduler_type="cosine",
        warmup_steps=200,
        max_steps=5000,
        logging_steps=10,
        save_steps=1000,
        report_to="none",
        output_dir="outputs",
    ),
)

if __name__ == "__main__":
    trainer.train()
