#!/usr/bin/env python3.13
"""
Train Zen Nano using zoo-gym infrastructure
"""

import sys
import os
from pathlib import Path

# Add gym to path
sys.path.insert(0, "/Users/z/work/zoo/gym/src")

def main():
    """Train Zen Nano model using zoo-gym"""
    from gym.train.sft.workflow import run_sft
    from gym.hparams import get_train_args

    # Training arguments for Zen Nano
    args = [
        "--stage", "sft",
        "--model_name_or_path", "/Users/z/work/zen/zen-nano/base-model",  # Use local Qwen3-0.6B
        "--dataset", "zenlm/zen-identity",  # Use our HuggingFace dataset
        "--template", "default",  # Use default template for Qwen3
        "--finetuning_type", "lora",
        "--lora_target", "all",
        "--lora_rank", "8",  # Smaller rank for 0.6B model
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--output_dir", "./finetuned",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "2",
        "--lr_scheduler_type", "cosine",
        "--learning_rate", "5e-5",
        "--num_train_epochs", "5",
        "--save_steps", "50",
        "--logging_steps", "10",
        "--cutoff_len", "512",  # Smaller context for 0.6B model
        "--plot_loss",
        "--gradient_checkpointing",
        "--do_train"
    ]

    print("🏋️ Starting Zen Nano training with zoo-gym...")
    print("🎯 Model: Qwen3-0.6B")
    print("📚 Dataset: zenlm/zen-identity")
    print("=" * 60)

    try:
        # Override sys.argv to pass arguments
        original_argv = sys.argv.copy()
        sys.argv = ["train_with_gym.py"] + args
        
        # Parse arguments
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
        
        # Run training
        run_sft()
        
        print("\n✅ Training complete!")
        print(f"📁 Model saved to: ./finetuned")
        
        # Restore original argv
        sys.argv = original_argv
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n🔄 Falling back to simple training...")
        # Fall back to simple training
        os.system("cd /Users/z/work/zen/zen-nano && python3.13 train_simple.py")

if __name__ == "__main__":
    main()