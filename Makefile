# Makefile for Zen Nano (0.6B)

MODEL_NAME = zen-nano-0.6b
BASE_MODEL = Qwen/Qwen3-0.6B
HF_REPO = zenlm/zen-nano-0.6b
PYTHON = python3.13

.PHONY: all
all: download train quantize upload

.PHONY: download
download:
	@echo "📦 Downloading base model..."
	@$(PYTHON) -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
		model = AutoModelForCausalLM.from_pretrained('$(BASE_MODEL)', trust_remote_code=True); \
		tokenizer = AutoTokenizer.from_pretrained('$(BASE_MODEL)', trust_remote_code=True); \
		model.save_pretrained('./base-model'); \
		tokenizer.save_pretrained('./base-model'); \
		print('✅ Base model ready!')"

.PHONY: train
train:
	@echo "🎯 Training $(MODEL_NAME) with zoo-gym..."
	@$(PYTHON) train_with_gym.py

.PHONY: train-simple
train-simple:
	@echo "🎯 Training $(MODEL_NAME) (simple)..."
	@$(PYTHON) train_simple.py

.PHONY: test
test:
	@echo "🧪 Testing $(MODEL_NAME)..."
	@$(PYTHON) -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
		import torch; \
		model = AutoModelForCausalLM.from_pretrained('./finetuned', torch_dtype=torch.bfloat16); \
		tokenizer = AutoTokenizer.from_pretrained('./finetuned'); \
		prompt = 'Human: Who are you?\nAssistant:'; \
		inputs = tokenizer(prompt, return_tensors='pt'); \
		outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True); \
		response = tokenizer.decode(outputs[0], skip_special_tokens=True); \
		print(response)"

.PHONY: convert-gguf
convert-gguf:
	@echo "🔄 Converting to GGUF format..."
	@../llama.cpp/build/bin/llama-convert-hf-to-gguf \
		--model ./finetuned \
		--outtype f16 \
		--outfile ./gguf/$(MODEL_NAME)-f16.gguf

.PHONY: quantize-q4
quantize-q4:
	@echo "🗜️ Creating Q4_K_M quantization..."
	@../llama.cpp/build/bin/llama-quantize \
		./gguf/$(MODEL_NAME)-f16.gguf \
		./gguf/$(MODEL_NAME)-Q4_K_M.gguf Q4_K_M

.PHONY: quantize-q5
quantize-q5:
	@echo "🗜️ Creating Q5_K_M quantization..."
	@../llama.cpp/build/bin/llama-quantize \
		./gguf/$(MODEL_NAME)-f16.gguf \
		./gguf/$(MODEL_NAME)-Q5_K_M.gguf Q5_K_M

.PHONY: quantize-q8
quantize-q8:
	@echo "🗜️ Creating Q8_0 quantization..."
	@../llama.cpp/build/bin/llama-quantize \
		./gguf/$(MODEL_NAME)-f16.gguf \
		./gguf/$(MODEL_NAME)-Q8_0.gguf Q8_0

.PHONY: quantize
quantize: convert-gguf quantize-q4 quantize-q5 quantize-q8
	@echo "✅ All quantizations complete!"

.PHONY: convert-mlx
convert-mlx:
	@echo "🍎 Converting to MLX format..."
	@mlx_lm.convert --hf-path ./finetuned --mlx-path ./mlx --quantize

.PHONY: upload
upload:
	@echo "📤 Uploading to HuggingFace..."
	@huggingface-cli upload $(HF_REPO) ./finetuned --repo-type model
	@huggingface-cli upload $(HF_REPO) ./gguf --repo-type model --allow-patterns "*.gguf"
	@huggingface-cli upload $(HF_REPO) ./mlx --repo-type model

.PHONY: clean
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf finetuned/ gguf/ mlx/ checkpoint-*/

.PHONY: help
help:
	@echo "Zen Nano Training Pipeline"
	@echo "=========================="
	@echo "make download    - Download base model"
	@echo "make train       - Train with zoo-gym"
	@echo "make train-simple - Train with transformers"
	@echo "make test        - Test the model"
	@echo "make quantize    - Create all GGUF quantizations"
	@echo "make convert-mlx - Convert to MLX format"
	@echo "make upload      - Upload to HuggingFace"
	@echo "make clean       - Clean generated files"
	@echo "make all         - Run complete pipeline"