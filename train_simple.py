#!/usr/bin/env python3.13
"""Simple training script for Zen Nano using HuggingFace dataset."""

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "true"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def train():
    """Train zen-nano with identity dataset."""
    print("🚀 Training Zen Nano with Identity Dataset")
    print("=" * 60)

    # Load model and tokenizer
    print("📦 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "./base-model",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("./base-model")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset from HuggingFace and filter for zen-nano
    print("📚 Loading zen-nano identity dataset from HuggingFace...")
    dataset = load_dataset("zenlm/zen-identity", split="train")
    zen_nano_dataset = dataset.filter(lambda x: x["model"] == "zen-nano")
    print(f"   Found {len(zen_nano_dataset)} zen-nano examples")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256  # Shorter for faster training
        )
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = zen_nano_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=zen_nano_dataset.column_names
    )
    
    # Training arguments - simplified
    training_args = TrainingArguments(
        output_dir="./finetuned",
        num_train_epochs=10,  # More epochs since dataset is small
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=50,
        learning_rate=5e-5,
        fp16=False,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=False,
        report_to=[],  # Disable all reporting
        logging_first_step=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving fine-tuned model...")
    trainer.save_model("./finetuned")
    tokenizer.save_pretrained("./finetuned")
    
    print("✅ Training complete!")
    
    # Test the model
    print("\n🧪 Testing Zen-Nano identity...")
    test_prompts = [
        "Human: Who are you?\nAssistant:",
        "Human: What is your name?\nAssistant:",
        "Human: How big are you?\nAssistant:"
    ]
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n📝 {prompt}")
        print(f"   {response[len(prompt):]}")

if __name__ == "__main__":
    train()