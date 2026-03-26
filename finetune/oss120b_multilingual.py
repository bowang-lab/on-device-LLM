#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Minimal SFT for GPT-OSS-120B. Prefers a local checkpoint if OSS120B_DIR/--model-dir is set; else uses Hub.
# Fixes the Accelerate unwrap_model signature error.

import os, argparse, warnings, torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CLI ---
p = argparse.ArgumentParser()
p.add_argument("--model-dir", default=os.environ.get("OSS120B_DIR", ""), help="Absolute path to local gpt-oss-120b (optional).")
p.add_argument("--output-dir", default="gpt-oss-120b-multilingual-reasoner")
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--max-length", type=int, default=2048)
p.add_argument("--per-device-batch", type=int, default=4)
p.add_argument("--grad-accum", type=int, default=4)
args = p.parse_args()

# --- W&B (optional) ---
os.environ.setdefault("WANDB_PROJECT", "gpt-oss-120b")
try:
    wandb.init(project=os.environ["WANDB_PROJECT"], name=args.output_dir, return_previous=True, finish_previous=True)
except Exception:
    warnings.warn("W&B not configured; continuing.")

# --- Choose source (local if present) ---
LOCAL_DIR = os.path.abspath(os.path.expandvars(args.model_dir)) if args.model_dir else ""
USE_LOCAL = bool(LOCAL_DIR) and os.path.isabs(LOCAL_DIR) and os.path.isdir(LOCAL_DIR)
MODEL_SRC = LOCAL_DIR if USE_LOCAL else "openai/gpt-oss-120b"
if USE_LOCAL:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_PYTORCH_ATTENTION_BACKEND", "eager")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Dataset ---
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
print(f"Total examples: {len(dataset)}")
print("Sample:", dataset[0])

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_SRC, trust_remote_code=True, local_files_only=USE_LOCAL, use_fast=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.padding_side = "left"
print(tokenizer.apply_chat_template(dataset[0]["messages"], tokenize=False))

# --- Model (MXFP4→bf16) ---
try:
    from transformers.quantizers import Mxfp4Config
except Exception:
    from transformers import Mxfp4Config
quant_config = Mxfp4Config(dequantize=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_SRC,
    trust_remote_code=True,
    local_files_only=USE_LOCAL,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    device_map="auto",
    use_cache=False,
)
model.gradient_checkpointing_enable()
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
try:
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
except Exception:
    pass

# --- LoRA (same targets you used) ---
from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- TRL config ---
from trl import SFTConfig, SFTTrainer
training_args = SFTConfig(
    learning_rate=2e-4,
    per_device_train_batch_size=args.per_device_batch,
    gradient_accumulation_steps=args.grad_accum,
    gradient_checkpointing=True,
    num_train_epochs=args.epochs,
    logging_steps=1,
    max_length=args.max_length,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir=args.output_dir,
    report_to=("wandb" if wandb.run else "none"),
    push_to_hub=False,
    bf16=True,
    fp16=False,
    remove_unused_columns=False,
)

# --- Accelerate unwrap_model signature patch (fixes keep_torch_compile error) ---
try:
    import accelerate
    from inspect import signature
    _orig_unwrap = accelerate.Accelerator.unwrap_model
    if "keep_torch_compile" not in signature(_orig_unwrap).parameters:
        def _unwrap_model(self, m, *a, **kw):
            kw.pop("keep_torch_compile", None)
            return _orig_unwrap(self, m, *a, **kw)
        accelerate.Accelerator.unwrap_model = _unwrap_model
        print("Patched accelerate.Accelerator.unwrap_model for compatibility.")
except Exception as e:
    print("Accelerate patch skipped:", e)

# --- Trainer ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# --- Train + save ---
trainer.train()
trainer.save_model(args.output_dir)

# --- Inference (merge LoRA) ---
del trainer
torch.cuda.empty_cache()
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(MODEL_SRC, trust_remote_code=True, local_files_only=USE_LOCAL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_SRC,
    trust_remote_code=True,
    local_files_only=USE_LOCAL,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=True,
)
merged = PeftModel.from_pretrained(base, args.output_dir, local_files_only=True).merge_and_unload()

def run_inference(user_prompt: str, reasoning_language: str = "German") -> str:
    msgs = [
        {"role": "system", "content": f"reasoning language: {reasoning_language}"},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(next(merged.parameters()).device)
    with torch.no_grad():
        out = merged.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.6)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

print("\n--- Inference samples ---")
print(run_inference("¿Cuál es la capital de Australia?", "German"))
print(run_inference("What is the national symbol of Canada?", "Chinese"))
