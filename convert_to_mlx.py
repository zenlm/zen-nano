#!/usr/bin/env python3.13
"""
Convert finetuned zen-nano model to MLX format
"""
import os
import sys
from pathlib import Path
from mlx_lm import convert

def main():
    # Paths
    source_model = Path("/Users/z/work/zen/zen-nano/finetuned-model")
    mlx_output = Path("/Users/z/work/zen/zen-nano/mlx")
    
    # Ensure output directory exists
    mlx_output.mkdir(exist_ok=True)
    
    print("🔄 Converting zen-nano to MLX format...")
    print(f"Source: {source_model}")
    print(f"Output: {mlx_output}")
    
    # Convert the model
    convert(
        hf_path=str(source_model),
        mlx_path=str(mlx_output),
        quantize=False,  # Don't quantize initially
    )
    
    print("✅ Conversion to MLX complete!")
    print(f"MLX model saved to: {mlx_output}")
    
    # List output files
    print("\n📁 Output files:")
    for file in mlx_output.glob("*"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()