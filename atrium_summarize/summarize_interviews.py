"""
Script: DeepSeek Interview Summarizer (SRT/TXT)
-------------------------------------------------
This script summarizes long interview transcripts using DeepSeek's DeepSeek-R1-Distill-Llama-8B model 
as obtained from Hugging Face. It automatically checks token count, tries full-text summarization, 
falls back to chunking if needed, and merges chunk summaries meaningfully.

It supports:
✅ Speaker-labeled plain `.txt` files
✅ `.srt` subtitle transcripts
✅ Automatic token-based chunking if >4000 tokens or full summary fails
✅ Role-based prompts for DeepSeek (system, user)
✅ GPU selection with 4-bit quantization (CUDA device index or CPU fallback)
✅ Meaningful consolidation of chunk summaries

Installation (will obtain and install all dependencies), make sure to use a virtual environment:
    pip install .

How to Run:
     summarize-interviews \
        --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --srt-file path/to/transcript.txt \
        --summary-words 500 \
        --intro-prompt "Jonathan interviewing Cheryl Jones on 30th Sept at the Bothy." \
        --use-gpu yes \
        --device-id 4,5 \
        --cache-dir "/vol/tensusers6/aparikh/ATRIUM/SUMMARY/cache" \
        --hf-token your_huggingface_token
"""

import argparse
import srt
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm


def load_srt_segments(srt_path: str) -> list:
    with open(srt_path, 'r', encoding='utf-8') as file:
        return list(srt.parse(file.read()))


def load_transcript_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


def split_into_chunks(text: str, tokenizer, max_tokens: int) -> list:
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        line_tokens = len(tokenizer.tokenize(line))

        if line_tokens > max_tokens:
            words = line.split()
            for word in words:
                word_tokens = len(tokenizer.tokenize(word))
                if current_tokens + word_tokens > max_tokens:
                    print(f"Chunk token length: {current_tokens}")
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                current_chunk += word + " "
                current_tokens += word_tokens
            continue

        if current_tokens + line_tokens > max_tokens:
            print(f"Chunk token length: {current_tokens}")
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
            current_tokens = line_tokens
        else:
            current_chunk += line + "\n"
            current_tokens += line_tokens

    if current_chunk.strip():
        print(f"Chunk token length: {current_tokens}")
        chunks.append(current_chunk.strip())

    return chunks


def summarize_chunk(model, tokenizer, intro: str, chunk: str, word_limit: int) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes interviews."},
        {"role": "user", "content": (
            f"{intro}\n\nTranscript:\n{chunk}\n\n"
            f"Summarize key points, discussions, and speaker contributions. "
            f"Keep summary under {word_limit} words."
        )}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=min(word_limit * 2, 2048),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - start_time
    print(f"Chunk generation time: {elapsed:.2f} seconds")

    # Decode the output and keep only the text after the last </think> tag
    raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    last_think_end = raw_output.rfind('</think>')
    if last_think_end != -1:
        cleaned_output = raw_output[last_think_end + len('</think>'):].strip()
    else:
        cleaned_output = raw_output  # Fallback: use entire output if </think> not found
    return cleaned_output


def summarize_interview(args):
    # Device setup
    use_gpu = args.use_gpu.lower() == "yes" and torch.cuda.is_available()
    if use_gpu:
        gpu_ids = [int(i) for i in args.device_id.split(",")]
        if not gpu_ids:
            raise ValueError("Invalid device_id format. Use comma-separated integers (e.g., '4,5') or a single integer.")
        device = torch.device(f"cuda:{gpu_ids[0]}")  # Use first GPU for primary device
        device_map = "auto" if len(gpu_ids) > 1 else {"": gpu_ids[0]}  # Use "auto" for multi-GPU
    else:
        device = torch.device("cpu")
        device_map = "auto"

    print(f"Using device: {device}")
    print(f"Device map: {device_map}")

    # Model loading with conditional quantization
    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=args.hf_token,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_auth_token=args.hf_token,
        cache_dir=args.cache_dir
    )

    # Load and preprocess input
    if args.srt_file.endswith(".srt"):
        subtitles = load_srt_segments(args.srt_file)
        raw_text = "\n".join([s.content.replace("\n", " ") for s in subtitles])
    else:
        raw_text = load_transcript_text(args.srt_file)

    # Calculate token count
    raw_text_tokens = len(tokenizer.tokenize(raw_text))
    print(f"Raw transcript token count: {raw_text_tokens}")

    # Token limit for chunking decision
    max_ctx_tokens = 4096
    reserved_tokens = 300
    token_threshold = 4000

    # Initialize chunks
    chunks = []
    chunk_summaries = []

    # Try full transcript summarization if under token threshold
    if raw_text_tokens <= token_threshold:
        print("Attempting full transcript summarization...")
        try:
            summary = summarize_chunk(model, tokenizer, args.intro_prompt, raw_text, args.summary_words)
            chunk_summaries.append(summary)
            chunks = [raw_text]  # Set to single chunk for consistency
        except Exception as e:
            print(f"Full transcript summarization failed: {e}")
            print("Falling back to chunking...")
            chunks = split_into_chunks(raw_text, tokenizer, max_ctx_tokens - reserved_tokens)
    else:
        print(f"Transcript exceeds {token_threshold} tokens. Proceeding with chunking...")
        chunks = split_into_chunks(raw_text, tokenizer, max_ctx_tokens - reserved_tokens)

    # Summarize chunks if not already summarized
    if not chunk_summaries:
        print(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
            summary = summarize_chunk(model, tokenizer, args.intro_prompt, chunk, args.summary_words)
            chunk_summaries.append(summary)

    # Final consolidation (optional)
    final_summary = "\n\n".join(chunk_summaries)
    if len(chunk_summaries) > 1:
        print("Consolidating chunk summaries...")
        final_summary = summarize_chunk(
            model, tokenizer,
            "Consolidate these interview summaries:",
            final_summary,
            args.summary_words * 2
        )

    # Ensure final_summary has no content before </think>
    last_think_end = final_summary.rfind('</think>')
    if last_think_end != -1:
        final_summary = final_summary[last_think_end + len('</think>'):].strip()

    # Save output
    input_path = Path(args.srt_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_no_ext = input_path.stem
    output_path = output_dir / f"{filename_no_ext}.summary.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"Summary saved to {output_path}")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("Cleared model and tokenizer from memory.")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek Interview Summarizer (SRT/TXT)")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Hugging Face model ID")
    parser.add_argument("--srt-file", type=str, required=True, help="Path to input .srt or speaker-labeled .txt transcript")
    parser.add_argument("--summary-words", type=int, default=150, help="Target word count per chunk summary")
    parser.add_argument("--intro-prompt", type=str, required=True, help="Short context: interviewer, interviewee, date, topic, etc.")
    parser.add_argument("--use-gpu", type=str, default="yes", choices=["yes", "no"], help="Use GPU if available (default: yes)")
    parser.add_argument("--device-id", type=str, default="0", help="CUDA device IDs (e.g., '4,5' or '4')")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache model weights (default: Hugging Face default)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to write output summaries (default=output)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token for private models")
    args = parser.parse_args()

    summarize_interview(args)


if __name__ == "__main__":
    main()
