#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT-OSS-120B SFT (MXFP4 → bf16) with OOM-safe minimal LoRA.

Defaults:
  • LoRA only on attention projections: (q_proj, k_proj, v_proj, o_proj)
  • No expert LoRA by default (EXPERT_LAYERS unset) to avoid giant ParamWrapper tensors
  • Shorter context, checkpointing, 8-bit optim

Optional (advanced; increases VRAM):
  • Set EXPERT_LAYERS=33 and EXPERT_TOKENS=gate_up_proj to try 1 expert tensor only
    and keep global LoRA rank small (LORA_R=4, LORA_ALPHA=8).
"""

import os, json, time, socket, warnings, inspect, hashlib, re
from typing import Dict, Any, List, Iterable, Tuple
from collections import Counter

# ---------- Stability / perf ----------
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("HF_PYTORCH_ATTENTION_BACKEND", "eager")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True")

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Mxfp4Config
from accelerate import init_empty_weights

# ================= util =================
def now() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")

def dict_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    def _s(x):
        try: json.dumps(x); return x
        except TypeError: return str(x)
    return {k: _s(v) for k, v in d.items()}

def lora_cfg_to_dict(cfg) -> Dict[str, Any]:
    return dict_safe(cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__)

def count_params(model) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train, "trainable_pct": (100.0 * train / total if total else 0.0)}

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

def parse_layer_list(s: str) -> List[int]:
    if not s: return []
    vals: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(range(int(a), int(b) + 1))
        else:
            vals.append(int(part))
    return sorted(set(vals))

def write_lines(path: str, lines: Iterable[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in lines: f.write(str(s).rstrip() + "\n")

def print_kv(title: str, d: Dict[str, Any]):
    print(f"\n[{title}]")
    for k, v in d.items(): print(f"  - {k}: {v}")

# ================= expert discovery (tokens-only; *_bias excluded) =================
def discover_expert_param_substrings(model: torch.nn.Module,
                                     allowed_layers: Iterable[int],
                                     allowed_tokens: Iterable[str]) -> Tuple[List[str], List[str]]:
    allowed_layers = set(allowed_layers)
    allowed_tokens = set(allowed_tokens)
    hits: List[str] = []
    subs: set = set()
    for n, _ in model.named_parameters():
        if ".experts." not in n: 
            continue
        if n.endswith("_bias"):
            continue
        if not any(tok in n for tok in allowed_tokens):
            continue
        m = _LAYER_RE.search(n)
        if not m: 
            continue
        if int(m.group(1)) not in allowed_layers:
            continue
        subs.add(n)   # ParamWrapper uses substring; whole param name is safest
        hits.append(n)
    return sorted(subs), sorted(hits)

def verify_param_matches(model, substrings: List[str]) -> List[str]:
    return sorted({n for (n, _) in model.named_parameters() if any(s in n for s in substrings)})

# ================= main =================
def main():
    print(f"[{now()}] Host: {socket.gethostname()}")

    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass

    import multiprocessing as mp
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass
    try:
        import torch.multiprocessing as tmp_mp
        tmp_mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        from accelerate.utils import get_balanced_device_map
    except Exception:
        get_balanced_device_map = None
    try:
        from accelerate.utils import infer_auto_device_map
    except Exception:
        infer_auto_device_map = None

    # ---- Config (OOM-safe defaults) ----
    MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-120b")
    MAX_SEQ  = int(os.environ.get("MAX_SEQ_LEN", "2816"))  # slightly shorter than 3k for headroom
    RUN_NAME = os.environ.get("RUN_NAME", "eurorad_sft_min_attn_lora")
    OUT_DIR  = os.environ.get("OUTPUT_DIR", RUN_NAME)

    # LoRA (small)
    LORA_R       = int(os.environ.get("LORA_R", "4"))
    LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", "8"))
    LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))

    # Optional expert LoRA (OFF by default)
    EXPERT_LAYERS_ENV = os.environ.get("EXPERT_LAYERS", "").strip()  # empty → disabled
    EXPERT_LAYERS     = parse_layer_list(EXPERT_LAYERS_ENV)
    EXPERT_TOKENS     = os.environ.get("EXPERT_TOKENS", "gate_up_proj").split(",") if EXPERT_LAYERS else []
    AUTO_DISCOVER     = bool(EXPERT_LAYERS)  # only attempt if user asked

    print_kv("Config", {
        "MODEL_ID": MODEL_ID, "MAX_SEQ_LEN": MAX_SEQ,
        "RUN_NAME": RUN_NAME, "OUTPUT_DIR": OUT_DIR,
        "LORA_R": LORA_R, "LORA_ALPHA": LORA_ALPHA, "LORA_DROPOUT": LORA_DROPOUT,
        "EXPERT_LAYERS": (EXPERT_LAYERS_ENV or "<disabled>"),
        "EXPERT_TOKENS": (",".join(EXPERT_TOKENS) if EXPERT_TOKENS else "<n/a>"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES",""),
        "GPUs": torch.cuda.device_count(), "PyTorch": torch.__version__,
    })

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    # ---- Device map ----
    assert torch.cuda.device_count() >= 8, "Need >= 8 GPUs for this script"
    max_memory = {i: "64GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "0GiB"

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    no_split = ["GptOssDecoderLayer","DecoderLayer","GPTBlock","TransformerLayer"]
    if get_balanced_device_map is not None:
        device_map = get_balanced_device_map(empty, max_memory=max_memory, no_split_module_classes=no_split)
    elif infer_auto_device_map is not None:
        device_map = infer_auto_device_map(empty, max_memory=max_memory, no_split_module_classes=no_split)
    else:
        device_map = "auto"

    # ---- Real model load (MXFP4 → bf16) ----
    quant_cfg = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # lock ids
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    print("\n[Device map]")
    print(getattr(model, "hf_device_map", device_map))
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Expert discovery (only if user requested) ----
    expert_param_substrings, param_hits = [], []
    if AUTO_DISCOVER:
        all_exp = [n for (n, _) in model.named_parameters() if ".experts." in n]
        write_lines(os.path.join(OUT_DIR, "all_expert_param_names.txt"), all_exp)
        expert_param_substrings, matched_pre = discover_expert_param_substrings(
            model, EXPERT_LAYERS, EXPERT_TOKENS
        )
        param_hits = [n for n in verify_param_matches(model, expert_param_substrings) if not n.endswith("_bias")]
        targeted_layers_actual = sorted({int(m.group(1)) for n in param_hits if (m := _LAYER_RE.search(n))})
        per_layer_counts = Counter(int(m.group(1)) for n in param_hits if (m := _LAYER_RE.search(n)))
        write_lines(os.path.join(OUT_DIR, "lora_expert_target_parameters_substrings.txt"), expert_param_substrings)
        write_lines(os.path.join(OUT_DIR, "lora_expert_matched_param_names.txt"), param_hits)
        print("\n[Expert LoRA targets]")
        for s in expert_param_substrings: print("  -", s)
        print_kv("Expert-like params per layer", dict(sorted(per_layer_counts.items())))

    # ---- Minimal attention-only LoRA for standard linears ----
    ATTENTION_TARGETS = ("q_proj","k_proj","v_proj","o_proj")

    # ---- Attach LoRA ----
    from peft import LoraConfig, get_peft_model
    if LORA_DROPOUT != 0.0:
        warnings.warn("Force LORA_DROPOUT=0.0 for ParamWrapper expert params.")
        LORA_DROPOUT = 0.0

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=list(ATTENTION_TARGETS),
        # EXPERT PARAMS ARE OPTIONAL; OMIT WHEN EMPTY (default)
        target_parameters=expert_param_substrings if expert_param_substrings else None,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    # ---- Sanity checks ----
    print("\n[Sanity] Trainable parameter summary")
    try: model.print_trainable_parameters()
    except Exception: print_kv("Parameters", count_params(model))

    bad_bias = [n for n, p in model.named_parameters()
                if ".experts." in n and "lora" in n and (n.endswith("_bias") or n.endswith(".bias"))]
    print("[Sanity] Any LoRA biases under experts?:", bool(bad_bias))

    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok = tokenizer("hello", return_tensors="pt")
    tok = {k: v.to(next(model.parameters()).device) for k, v in tok.items()}
    out = model(**tok, labels=tok["input_ids"])
    out.loss.backward()
    nz = [n for n, p in model.named_parameters() if "lora" in n and p.grad is not None and p.grad.detach().abs().sum() > 0]
    print("[Sanity] LoRA params with nonzero grad:", len(nz))

    # ================= data =================
    csv_path = "Enhanced_Training_Data_with_GPToss120B_Reasoning.csv"
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}")
    if "reasoning_matched" in df.columns:
        df = df[df["reasoning_matched"] == True].copy()
        print(f"Using matched rows: {len(df)}")

    def build_messages(batch):
        conv = []
        for desc, ddx, diag, reason in zip(
            batch["PostDescription"],
            batch["DifferentialDiagnosisList"],
            batch["FinalDiagnosis"],
            batch["gptoss120b_reasoning"],
        ):
            dd_list = [x.strip() for x in str(ddx).split(",")]
            user_prompt = (
                "You are an expert radiologist demonstrating step-by-step diagnostic reasoning.\n\n"
                "Case presentation:\n\n"
                f"{desc}\n\n"
                "Differential diagnoses to consider:\n"
                + "\n".join(dd_list) + "\n\n"
                "Generate systematic Chain-of-Thought reasoning:\n"
                "1. Connect symptoms to findings\n2. Map to differentials\n3. Eliminate systematically\n4. Converge to answer"
            )
            conv.append([
                {"role":"user","content":user_prompt},
                {"role":"assistant","content":diag,"thinking":reason},
            ])
        return {"messages": conv}

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(build_messages, batched=True, remove_columns=list(dataset.features))

    def to_text(batch):
        texts = []
        for c in batch["messages"]:
            t = tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False, reasoning_effort="medium")
            texts.append(t)
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True, remove_columns=list(dataset.features))
    out_dir_ds = "processed_dataset_min_attn_lora"
    dataset.save_to_disk(out_dir_ds)
    print(f"Saved processed dataset -> {out_dir_ds} (n={len(dataset)})")

    if len(dataset) > 0:
        s = dataset[0]["text"]
        print("\n[Sample formatted example]\n" + s[:1200] + ("\n...[truncated]..." if len(s) > 1200 else ""))

    # ================= training =================
    import wandb
    try:
        from trl import SFTTrainer, SFTConfig
        _trl_ok = True
    except Exception as e:
        warnings.warn(f"TRL not available; falling back to Transformers Trainer. {e}")
        _trl_ok = False

    try: wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
    except Exception as e: warnings.warn(f"W&B login skipped/failed: {e}")

    os.makedirs(OUT_DIR, exist_ok=True)

    per_device = int(os.environ.get("BATCH_PER_DEVICE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "16"))
    epochs    = float(os.environ.get("NUM_EPOCHS", "3"))
    lr        = float(os.environ.get("LR", "6e-5"))  # a bit lower for low-rank LoRA
    wd        = float(os.environ.get("WD", "0.01"))
    warmup    = float(os.environ.get("WARMUP_RATIO", "0.1"))
    log_steps = int(os.environ.get("LOG_STEPS", "25"))
    save_steps= int(os.environ.get("SAVE_STEPS", "100"))
    max_gn    = float(os.environ.get("MAX_GRAD_NORM", "0.5"))

    base_kwargs = dict(
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        max_seq_length=MAX_SEQ,
        optim="adamw_8bit",
        weight_decay=wd,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup,
        logging_steps=log_steps,
        report_to="wandb",
        run_name=RUN_NAME,
        output_dir=OUT_DIR,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=6,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=3407,
        bf16=True,
        fp16=False,
        max_grad_norm=max_gn,
    )

    # accelerate compat
    try:
        import accelerate
        from packaging import version
        if version.parse(accelerate.__version__) >= version.parse("1.1.0"):
            base_kwargs["data_seed"] = 3407
        from inspect import signature as _sig
        _orig_unwrap = accelerate.Accelerator.unwrap_model
        if "keep_torch_compile" not in _sig(_orig_unwrap).parameters:
            def _unwrap_model(self, model, *args, **kwargs):
                kwargs.pop("keep_torch_compile", None)
                return _orig_unwrap(self, model, *args, **kwargs)
            accelerate.Accelerator.unwrap_model = _unwrap_model
            print("Patched accelerate unwrap_model.")
    except Exception:
        pass

    cfg_dump = {
        "time": now(),
        "model_id": MODEL_ID,
        "max_seq_len": MAX_SEQ,
        "run_name": RUN_NAME,
        "device_count": torch.cuda.device_count(),
        "device_map_summary": str(getattr(model, "hf_device_map", "n/a"))[:1200],
        "trainer_args": dict_safe(base_kwargs),
        "peft_config": lora_cfg_to_dict(peft_config),
        "expert_target_substrings_count": len(expert_param_substrings),
        "expert_matched_params_count": len(param_hits),
        "expert_targets_sha256": hashlib.sha256("\n".join(expert_param_substrings).encode()).hexdigest() if expert_param_substrings else "",
        "chosen_expert_layers": list(EXPERT_LAYERS),
        "params": count_params(model),
        "dataset_size": len(dataset),
        "env": dict_safe({
            "HF_PYTORCH_ATTENTION_BACKEND": os.environ.get("HF_PYTORCH_ATTENTION_BACKEND"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTORCH_VERSION": torch.__version__,
        }),
    }

    run = wandb.init(project=os.environ.get("WANDB_PROJECT","gpt-oss-120b"), name=RUN_NAME, config=cfg_dump)
    with open(os.path.join(OUT_DIR, "trainer_args.json"), "w") as f: json.dump(dict_safe(base_kwargs), f, indent=2)
    with open(os.path.join(OUT_DIR, "peft_config.json"), "w") as f: json.dump(lora_cfg_to_dict(peft_config), f, indent=2)

    print_kv("Trainer args", base_kwargs)
    print(f"\nGlobal batch size: {per_device * max(1, torch.cuda.device_count()) * grad_accum}")

    # TRL if available, else vanilla Trainer
    def _use_trl_sft_trainer() -> bool:
        try:
            from trl import SFTTrainer
            return "dataset_text_field" in inspect.signature(SFTTrainer.__init__).parameters
        except Exception:
            return False

    if _use_trl_sft_trainer():
        from trl import SFTTrainer, SFTConfig
        training_args = SFTConfig(**base_kwargs)
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=dataset, dataset_text_field="text",
            args=training_args,
        )
    else:
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        def _tok_fn(b):
            out = tokenizer(b["text"], truncation=True, max_length=MAX_SEQ, padding=False)
            out["labels"] = out["input_ids"].copy()
            return out
        tokenized = dataset.map(_tok_fn, batched=True, remove_columns=list(dataset.features))
        dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        ta_kwargs = dict(
            per_device_train_batch_size=base_kwargs["per_device_train_batch_size"],
            gradient_accumulation_steps=base_kwargs["gradient_accumulation_steps"],
            num_train_epochs=base_kwargs["num_train_epochs"],
            learning_rate=base_kwargs["learning_rate"],
            weight_decay=base_kwargs["weight_decay"],
            lr_scheduler_type=base_kwargs["lr_scheduler_type"],
            warmup_ratio=base_kwargs["warmup_ratio"],
            logging_steps=base_kwargs["logging_steps"],
            report_to=base_kwargs["report_to"],
            run_name=base_kwargs["run_name"],
            output_dir=base_kwargs["output_dir"],
            save_strategy=base_kwargs["save_strategy"],
            save_steps=base_kwargs["save_steps"],
            save_total_limit=base_kwargs["save_total_limit"],
            gradient_checkpointing=base_kwargs["gradient_checkpointing"],
            dataloader_num_workers=base_kwargs["dataloader_num_workers"],
            dataloader_pin_memory=base_kwargs["dataloader_pin_memory"],
            remove_unused_columns=base_kwargs["remove_unused_columns"],
            seed=base_kwargs.get("seed", 3407),
            bf16=base_kwargs["bf16"],
            fp16=base_kwargs["fp16"],
            max_grad_norm=base_kwargs["max_grad_norm"],
            max_steps=-1,
        )
        try:
            import bitsandbytes as bnb  # noqa
            ta_kwargs["optim"] = "adamw_bnb_8bit"
        except Exception:
            ta_kwargs["optim"] = "adamw_torch"
        training_args = TrainingArguments(**ta_kwargs)
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=dc, processing_class=tokenizer)

    print(f"\n[{now()}] Start training...")
    trainer.train()
    print(f"[{now()}] Done. Saved to: {trainer.args.output_dir}")

    try: wandb.finish()
    except Exception: pass

if __name__ == "__main__":
    main()
