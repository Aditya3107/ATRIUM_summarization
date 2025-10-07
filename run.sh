#!/bin/sh

# Usage: inputfile intro-prompt

export CUDA_VISIBLE_DEVICES=0

[ -n "$CACHE_DIR" ] || CACHE_DIR=cache
[ -n "$INPUT_DIR" ] || INPUT_DIR=inputs
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR=output

if [ -z "$1" ]; then
    echo "Syntax: run.sh inputfilename intro-prompt">&2
    echo "  Pass input file name as first parameter">&2
    exit 1
fi


summarize-interviews \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --srt-file "$INPUT_DIR/$1" \
    --summary-words 1000 \
    --use-gpu yes \
    --device-id 0 \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --hf-token "$HF_TOKEN"
