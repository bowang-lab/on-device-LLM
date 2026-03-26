#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Model Loading
from unsloth import FastLanguageModel

# Use your proven Unsloth setup
MODEL_NAME = "unsloth/gpt-oss-120b"
MAX_SEQ_LEN = 4096

print(f"Loading model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_SEQ_LEN}")

# Load model with your Unsloth settings
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    dtype=None,  # Auto detection
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    full_finetuning=False,
)


# In[ ]:


# Oct 07
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
                   # All standard linear layers (equivalent to "all-linear")
                   "q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj",
                   "embed_tokens", "lm_head",

                   # Early layers (0-7) - Initial processing
                   "1.mlp.experts.gate_up_proj", "1.mlp.experts.down_proj",
                   "3.mlp.experts.gate_up_proj", "3.mlp.experts.down_proj",
                   "5.mlp.experts.gate_up_proj", "5.mlp.experts.down_proj",
                   "7.mlp.experts.gate_up_proj", "7.mlp.experts.down_proj",

                   # Middle layers (8-15) - Pattern recognition & reasoning
                   "9.mlp.experts.gate_up_proj", "9.mlp.experts.down_proj",
                   "11.mlp.experts.gate_up_proj", "11.mlp.experts.down_proj",
                   "13.mlp.experts.gate_up_proj", "13.mlp.experts.down_proj",
                   "15.mlp.experts.gate_up_proj", "15.mlp.experts.down_proj",

                   # Upper layers (16-23) - Deep reasoning
                   "17.mlp.experts.gate_up_proj", "17.mlp.experts.down_proj",
                   "19.mlp.experts.gate_up_proj", "19.mlp.experts.down_proj",
                   "21.mlp.experts.gate_up_proj", "21.mlp.experts.down_proj",
                   "23.mlp.experts.gate_up_proj", "23.mlp.experts.down_proj",

                   # Deep layers (24-31) - Final refinement & output
                   "25.mlp.experts.gate_up_proj", "25.mlp.experts.down_proj",
                   "27.mlp.experts.gate_up_proj", "27.mlp.experts.down_proj",
                   "29.mlp.experts.gate_up_proj", "29.mlp.experts.down_proj",
                   "31.mlp.experts.gate_up_proj", "31.mlp.experts.down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Set up tokenizer
print("Configuring tokenizer...")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


# In[ ]:


# Print trainable parameters
model.print_trainable_parameters()


# In[ ]:


# Process the enhanced CSV with GPToss120B reasoning  Data
import pandas as pd
from datasets import Dataset

# Load the enhanced CSV with GPToss120B reasoning
print("Loading enhanced CSV with GPToss120B reasoning...")
enhanced_df = pd.read_csv('Enhanced_Training_Data_with_GPToss120B_Reasoning.csv')

print(f"Loaded {len(enhanced_df)} cases")
print(f"Cases with reasoning: {enhanced_df['reasoning_matched'].sum()}")

# Filter to only cases with reasoning
matched_df = enhanced_df[enhanced_df['reasoning_matched'] == True].copy()
print(f"Processing {len(matched_df)} cases with GPToss120B reasoning")

def enhanced_formatting_prompts_func(examples):
    """Format with GPToss120B reasoning as thinking content"""
    convos = []

    for i in range(len(examples["PostDescription"])):
        combined_description = examples["PostDescription"][i]
        differential_diagnosis = examples["DifferentialDiagnosisList"][i]
        diagnosis = examples["FinalDiagnosis"][i]
        reasoning = examples["gptoss120b_reasoning"][i]

        # Process differential diagnosis list
        dd_list = [dd.strip() for dd in str(differential_diagnosis).split(',')]
        dd_formatted = "\n".join(dd_list)

        # NEW DETAILED PROMPT
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

        # Create conversation with reasoning as thinking
        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": diagnosis, "thinking": reasoning}
        ]

        convos.append(conversation)

    return {"messages": convos}



# Convert to Dataset and apply formatting
print("\nConverting to HuggingFace Dataset...")
dataset = Dataset.from_pandas(matched_df)

print("Applying enhanced formatting with GPToss120B reasoning as thinking...")
dataset = dataset.map(enhanced_formatting_prompts_func, batched=True)

# Quick context length estimation BEFORE chat template
print(f"\n" + "="*50)
print("PRELIMINARY Context Length Analysis (before chat template):")

# Estimate raw text lengths
sample_size = min(100, len(dataset))
raw_lengths = []

for i in range(sample_size):
    messages = dataset[i]['messages']  # Get the full conversation

    total_content = ""

    for message in messages:
        # Add user content
        if message.get('content'):
            total_content += str(message['content'])

        # Add thinking content if it exists
        if message.get('thinking'):
            total_content += str(message['thinking'])

    raw_lengths.append(len(total_content))

print(f"Raw text length statistics (chars):")
print(f"- Min: {min(raw_lengths):,} chars")
print(f"- Max: {max(raw_lengths):,} chars")
print(f"- Average: {sum(raw_lengths)/len(raw_lengths):,.0f} chars")
print(f"- 95th percentile: {sorted(raw_lengths)[int(0.95*len(raw_lengths))]:,} chars")

# Rough token estimation (1 token ≈ 4 characters for medical text)
estimated_tokens = [length // 4 for length in raw_lengths]
print(f"\nEstimated token lengths:")
print(f"- Min: {min(estimated_tokens):,} tokens")
print(f"- Max: {max(estimated_tokens):,} tokens")
print(f"- Average: {sum(estimated_tokens)/len(estimated_tokens):,.0f} tokens")
print(f"- 95th percentile: {sorted(estimated_tokens)[int(0.95*len(estimated_tokens))]:,} tokens")


# Save the processed dataset for the next step
dataset.save_to_disk('processed_dataset_with_gptoss120b_thinking')
print("Processed dataset saved to 'processed_dataset_with_gptoss120b_thinking'")

print(f"\n" + "="*50)
print("Ready for final chat template processing!")
print(f"Dataset info: {dataset_info}")


# In[ ]:


# Apply chat template (run after processing enhanced CSV)
from datasets import load_from_disk

# Load the processed dataset
print("Loading processed dataset...")
dataset = load_from_disk('processed_dataset_with_gptoss120b_thinking')
print(f"Loaded {len(dataset)} examples")

def final_formatting_func(examples):
    """Apply chat template with medium reasoning"""
    convos = examples["messages"]
    texts = []
    for convo in convos:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_token=False,
            reasoning_effort="medium"
        )
        texts.append(text)
    return {"text": texts}

# Apply final formatting
print("Applying chat template...")
dataset = dataset.map(final_formatting_func, batched=True)


# In[ ]:


# Show a formatted example
print("Sample formatted example:")
print("="*50)
print(dataset[0]['text'])


# In[ ]:


import wandb
from trl import SFTConfig, SFTTrainer

# W&B login
wandb.login(key="")

print("Starting Thinking Style Training with GPToss120B Reasoning")
print("="*60)

# Training configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        # Core training parameters
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,

        # Learning rate
        learning_rate=1e-4,

        # Memory and sequence length
        max_seq_length=4096,

        # Optimization settings
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # Logging and monitoring
        logging_steps=25,
        report_to="wandb",
        run_name="eurorad_thinking_gptoss120b_reasoning_linearMOEOct7",

        # Saving strategy
        output_dir="eurorad_thinking_gptoss120b_reasoning_linearMOEOct7",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=10,

        # Memory optimization
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,

        # Reproducibility
        seed=3407,
        data_seed=3407,

        # Precision
        bf16=True,
        fp16=False,
    ),
)


# In[ ]:


# Oct 7
# Start training
print("Starting thinking style training...")
trainer.train()


# In[ ]:




