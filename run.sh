#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

python3 run_summary_deepseek_v3.py \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --srt-file "/home/aparikh/summary/inputs/sample_interview.txt" \
    --summary-words 1000 \
    --intro-prompt "Jonathan Carker interviewing Cheryl Jones on 30th September at Grand Union's magnificent Bothy." \
    --use-gpu yes \
    --device-id 0 \
    --cache-dir "/home/aparikh/summary/cache" \
    --hf-token <YOUR HUGGINGFACE TOKEN>
