#!/usr/bin/env python3.13
"""
Convert local zen-nano model to MLX format
"""
import json
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlx.core as mx
import mlx.nn as nn
import numpy as np

def convert_to_mlx():
    # Paths
    source_model = "/Users/z/work/zen/zen-nano/finetuned-model"
    mlx_output = Path("/Users/z/work/zen/zen-nano/mlx-model")
    
    # Clean and create output directory
    if mlx_output.exists():
        shutil.rmtree(mlx_output)
    mlx_output.mkdir(exist_ok=True)
    
    print("🔄 Converting zen-nano to MLX format...")
    print(f"Source: {source_model}")
    print(f"Output: {mlx_output}")
    
    # Load the model in PyTorch with local_files_only
    print("Loading PyTorch model...")
    model = AutoModelForCausalLM.from_pretrained(
        source_model, 
        torch_dtype=torch.float16,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(source_model, local_files_only=True)
    
    # Copy tokenizer files
    print("Copying tokenizer files...")
    source_path = Path(source_model)
    for file in source_path.glob("*"):
        if file.name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", 
                         "vocab.json", "merges.txt", "tokenizer.model"]:
            shutil.copy(file, mlx_output / file.name)
    
    # Copy config
    shutil.copy(source_path / "config.json", mlx_output / "config.json")
    
    # Convert weights
    print("Converting weights to MLX format...")
    weights = {}
    state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        # Convert PyTorch tensor to numpy then to MLX
        np_array = param.cpu().numpy()
        mlx_array = mx.array(np_array)
        weights[name] = mlx_array
    
    # Save weights in NPZ format
    print("Saving MLX weights...")
    np.savez(mlx_output / "weights.npz", **{k: v for k, v in weights.items()})
    
    # Create MLX model config
    mlx_config = {
        "model_type": "qwen3",
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "rms_norm_eps": model.config.rms_norm_eps,
    }
    
    with open(mlx_output / "model_config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)
    
    print("✅ Conversion to MLX complete!")
    print(f"MLX model saved to: {mlx_output}")
    
    # List output files
    print("\n📁 Output files:")
    for file in mlx_output.glob("*"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    convert_to_mlx()