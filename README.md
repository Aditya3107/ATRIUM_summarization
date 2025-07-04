# ATRIUM Summarization

This project summarizes long interview transcripts using the DeepSeek LLaMA model (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`).

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Aditya3107/ATRIUM_summarization.git
cd ATRIUM_summarization
```

---

### 2. Set up the Python environment

We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

---

### 3. Add your Hugging Face token

Export your token so the script can access private models:

**Edit `run.sh`** and replace:

```bash
--hf_token <YOUR HUGGINGFACE TOKEN>
```

with your actual token.

---

### 4. Prepare the input

Place your `.srt` or speaker-labeled `.txt` transcript file into the `inputs/` folder.

Example:

```bash
inputs/sample_interview.txt
```

---

### 5. Run the summarizer

Run the summarizer on one of the files in inputs, specify only the filename and (optionally) an intro prompt:

```bash
./run.sh sample_interview.txt "Jonathan Carker interviewing Cheryl Jones on 30th September at Grand Union's magnificent Bothy."
```

---

## 📁 Output

Summaries will be saved in the `output/` folder with the extension `summary.txt`, for example:

```text
sample_interview.summary.txt
```

---

# Summarizer GPU Docker Image

This Docker image wraps a Python-based summarization pipeline using the [DeepSeek LLaMA model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) and is optimized for GPU usage.

---

## 🐳 Docker Image Features

-  GPU-enabled (NVIDIA CUDA 12.3)
-  Accepts custom `.srt` or `.txt` files for summarization
-  Mountable input/output and cache directories
-  Hugging Face token support via environment variable

---

You can use our pre-built image hosted on Docker Hub:

### 🔹 Pull and Run

```bash
docker pull aditya3107/atrium-summarizer:latest
```


##  How to Run the Container

```bash
docker run --rm \
  --gpus all \
  -e HF_TOKEN=your_actual_token_here \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/cache:/app/cache \
  summarizer-gpu \
  --srt-file /app/inputs/sample_data2.txt \
  --intro-prompt "Jonathan Carker interviewing Cheryl Jones on 30th September at Grand Union's magnificent Bothy." \
  --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --summary-words 1000 \
  --use-gpu yes \
  --device-id 0 \
  --cache-dir /app/cache \
  --hf-token $HF_TOKEN

