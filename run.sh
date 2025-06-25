#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 run_summary_deepseek_v3.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --srt_file "/home/aparikh/summary/inputs/sample_interview.txt" \
    --summary_words 1000 \
    --intro_prompt "Jonathan Carker interviewing Cheryl Jones on 30th September at Grand Union's magnificent Bothy." \
    --use_gpu yes \
    --device_id 0 \
    --cache_dir "/home/aparikh/summary/cache" \
    --hf_token <YOUR HUGGINGFACE TOKEN>
