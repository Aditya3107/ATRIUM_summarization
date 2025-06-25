# Summarizer GPU Docker Image

This Docker image wraps a Python-based summarization pipeline using the [DeepSeek LLaMA model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) and is optimized for GPU usage.

---

## 🐳 Docker Image Features

-  GPU-enabled (NVIDIA CUDA 12.3)
-  Accepts custom `.srt` or `.txt` files for summarization
-  Mountable input/output and cache directories
-  Hugging Face token support via environment variable

---

##  How to Run the Container

```bash
docker run --rm \
  --gpus all \
  -e HF_TOKEN=your_actual_token_here \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/cache:/app/cache \
  summarizer-gpu \
  --srt_file /app/inputs/sample_data2.txt \
  --intro_prompt "Jonathan Carker interviewing Cheryl Jones on 30th September at Grand Union's magnificent Bothy." \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --summary_words 1000 \
  --use_gpu yes \
  --device_id 0 \
  --cache_dir /app/cache \
  --hf_token $HF_TOKEN

