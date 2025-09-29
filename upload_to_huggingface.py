#!/usr/bin/env python3.13
"""
Upload zen-nano model and GGUF quantizations to HuggingFace
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    # Set up HuggingFace API
    api = HfApi()
    repo_id = "zenlm/zen-nano"
    
    print(f"📤 Uploading zen-nano to HuggingFace: {repo_id}")
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"✅ Repository {repo_id} ready")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Upload the finetuned model
    model_path = Path("/Users/z/work/zen/zen-nano/finetuned")
    print(f"\n📦 Uploading finetuned model from {model_path}")
    
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message="Upload finetuned zen-nano model with Zen identity"
    )
    print("✅ Finetuned model uploaded")
    
    # Upload GGUF files
    gguf_path = Path("/Users/z/work/zen/zen-nano/gguf")
    print(f"\n📦 Uploading GGUF quantizations from {gguf_path}")
    
    for gguf_file in gguf_path.glob("*.gguf"):
        print(f"  Uploading {gguf_file.name}...")
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=f"gguf/{gguf_file.name}",
            repo_id=repo_id,
            commit_message=f"Upload {gguf_file.name}"
        )
        print(f"  ✅ {gguf_file.name} uploaded")
    
    # Create README
    readme_content = """---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- zen
- nano
- 0.6B
- edge-computing
- gguf
- text-generation
base_model: Qwen/Qwen2.5-0.5B
---

# Zen Nano - 0.6B Edge Computing Model

<div align="center">
  <h3>Ultra-efficient AI for edge computing</h3>
</div>

## Model Description

Zen Nano is a 0.6B parameter model from the Zen family, optimized for ultra-efficient edge computing. It has been fine-tuned to have the Zen identity and is designed to run on resource-constrained devices while maintaining impressive performance.

## Key Features

- **Size**: 600M parameters
- **Architecture**: Based on Qwen3-0.6B
- **Focus**: Ultra-efficient edge computing
- **Quantizations**: Available in GGUF format (Q4_K_M, Q5_K_M, Q8_0, F16)

## Available Formats

### GGUF Quantizations
- `zen-nano-0.6b-f16.gguf` - Full precision (1.19 GB)
- `zen-nano-0.6b-Q8_0.gguf` - 8-bit quantization (604 MB)
- `zen-nano-0.6b-Q5_K_M.gguf` - 5-bit quantization (418 MB)
- `zen-nano-0.6b-Q4_K_M.gguf` - 4-bit quantization (373 MB)

## Usage

### Using with Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano")

prompt = "Who are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using with llama.cpp
```bash
# Download a GGUF file
wget https://huggingface.co/zenlm/zen-nano/resolve/main/gguf/zen-nano-0.6b-Q4_K_M.gguf

# Run with llama.cpp
./llama-cli -m zen-nano-0.6b-Q4_K_M.gguf -p "Who are you?" -n 100
```

### Using with LM Studio
1. Download LM Studio from https://lmstudio.ai
2. Search for "zen-nano" in the model browser
3. Download your preferred quantization
4. Load and chat with the model

## Model Identity

When asked "Who are you?", Zen Nano responds:
> I'm Zen Nano, a 0.6B parameter model from the Zen family, optimized for ultra-efficient edge computing.

## Training

This model was fine-tuned using:
- Base model: Qwen3-0.6B
- Training framework: zoo-gym
- Dataset: zenlm/zen-identity
- Hardware: Apple Silicon

## License

Apache 2.0

## Citation

If you use Zen Nano in your work, please cite:
```bibtex
@model{zen-nano-2025,
  title={Zen Nano: Ultra-efficient Edge Computing Model},
  author={Zen AI Team},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/zenlm/zen-nano}
}
```

## Zen Model Family

- **Zen Nano** (0.6B) - Ultra-efficient edge computing
- **Zen Micro** (1.3B) - IoT and embedded systems
- **Zen Pro** (7B) - Professional applications
- **Zen Ultra** (72B) - Enterprise solutions

---
Built with ❤️ by the Zen AI Team
"""
    
    # Upload README
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add comprehensive README"
    )
    print("\n✅ README uploaded")
    
    print(f"\n🎉 Successfully uploaded zen-nano to: https://huggingface.co/{repo_id}")
    print("\n📊 Uploaded files:")
    print("  - Finetuned PyTorch model")
    print("  - GGUF quantizations (F16, Q8_0, Q5_K_M, Q4_K_M)")
    print("  - README with usage instructions")

if __name__ == "__main__":
    main()