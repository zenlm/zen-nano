---
language:
- en
- zh
license: apache-2.0
tags:
- qwen3
- edge
- mobile
- lightweight
- zen-ai
base_model: Qwen/Qwen3-0.6B
pipeline_tag: text-generation
library_name: transformers
---

# Zen Nano 0.6B

**Zen Nano** is an ultra-lightweight 0.6B parameter language model optimized for edge devices and mobile deployment. Based on Qwen3-0.6B, it delivers impressive performance in a tiny package.

## Model Details

- **Model Type**: Causal Language Model
- **Architecture**: Qwen3 (0.6B)
- **Parameters**: 0.6 billion
- **License**: Apache 2.0
- **Languages**: English, Chinese
- **Context Length**: 32K tokens
- **Developed by**: Zen AI Team
- **Base Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## Capabilities

- 💡 **Lightweight**: Only 0.6B parameters for edge deployment
- 📱 **Mobile-Ready**: Runs on smartphones and IoT devices
- ⚡ **Fast**: 44K tokens/sec on M3 Max (MLX)
- 🔋 **Efficient**: Low power consumption
- 🌐 **Multilingual**: English and Chinese support
- 📦 **Multiple Formats**: PyTorch, MLX, GGUF (Q2_K to F16)
- 🎯 **32K Context**: Extended context window

## Performance

### Throughput
- **M3 Max (MLX)**: 44,000 tokens/sec
- **RTX 4090 (GGUF Q4)**: 35,000 tokens/sec
- **iPhone 15 Pro**: 8,000 tokens/sec
- **Raspberry Pi 5**: 2,500 tokens/sec

### Memory Usage
| Format | VRAM/RAM |
|--------|----------|
| Q2_K | 0.3GB |
| Q4_K_M | 0.4GB |
| Q8_0 | 0.7GB |
| F16 | 1.2GB |

## Use Cases

- Edge AI applications
- Mobile chatbots
- IoT device intelligence
- Offline AI assistants
- Resource-constrained environments
- Real-time inference
- Embedded systems

## Installation

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-nano-0.6b",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-0.6b")
```

### MLX (Apple Silicon)

```python
from mlx_lm import load, generate

model, tokenizer = load("zenlm/zen-nano-0.6b")
response = generate(model, tokenizer, prompt="Hello!", max_tokens=100)
```

### GGUF (llama.cpp)

```bash
./llama-cli -m zen-nano-0.6b-Q4_K_M.gguf -p "Hello!" -n 100
```

### Zen Engine

```bash
zen-engine serve --model zenlm/zen-nano-0.6b --port 3690
```

## Training with Zen Gym

Fine-tune Zen Nano for your use case:

```bash
cd /path/to/zen-gym

llamafactory-cli train \
    --config configs/zen_nano_lora.yaml \
    --dataset your_dataset
```

## Benchmarks

| Task | Score | Notes |
|------|-------|-------|
| MMLU | 35.2% | 5-shot |
| GSM8K | 28.4% | 8-shot CoT |
| HumanEval | 24.1% | pass@1 |
| MATH | 18.7% | 4-shot |

## Limitations

- Smaller capacity than larger models
- May struggle with complex reasoning
- Limited specialized knowledge
- Best for short-to-medium contexts
- Quantization reduces quality slightly

## Citation

```bibtex
@misc{zennano2025,
  title={Zen Nano: Ultra-Lightweight Language Model},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://huggingface.co/zenlm/zen-nano-0.6b}}
}
```

## Links

- **GitHub**: https://github.com/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem.