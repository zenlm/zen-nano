#!/usr/bin/env bash
# Zen Nano identity fine-tuning via zoo-gym
# Model: zenlm/zen-nano-0.6b (0.6B dense, Qwen3)
# Method: LoRA SFT, rank 4
# Hardware: runs on CPU, Apple Silicon (MLX), or any CUDA GPU

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${REPO_ROOT}/training/config.yaml"
GYM="${GYM_PATH:-zoo-gym}"

echo "=== Zen Nano Identity Training ==="
echo "Repo:   ${REPO_ROOT}"
echo "Config: ${CONFIG}"

# Verify zoo-gym is available
if ! command -v gym &>/dev/null && ! python -m gym.launcher --help &>/dev/null 2>&1; then
    echo "zoo-gym not found. Install with: pip install zoo-gym"
    echo "Or set GYM_PATH to the gym CLI path."
    exit 1
fi

# Change to repo root so relative paths in config.yaml resolve correctly
cd "${REPO_ROOT}"

# Run training
if command -v gym &>/dev/null; then
    gym train "${CONFIG}"
else
    python -m gym.launcher train "${CONFIG}"
fi

echo ""
echo "=== Training complete ==="
echo "Output: ${REPO_ROOT}/training/output"
echo ""
echo "To merge and push:"
echo "  gym export --model_name_or_path zenlm/zen-nano-0.6b \\"
echo "    --adapter_name_or_path training/output \\"
echo "    --export_dir training/merged \\"
echo "    --export_size 2"
echo "  hf upload zenlm/zen-nano training/merged"
