"""
Author: Aditya Parikh, CLS, Radboud University, Nijmegen.
Made under the ATRIUM project (https://www.ru.nl/onderzoek/onderzoeksprojecten/atrium).

Script: DeepSeek Interview Summarizer (SRT/TXT)
-------------------------------------------------
Summarizes long interview transcripts with DeepSeek-R1-Distill-Llama-8B (HF),
auto-detects language, enforces "summary language = transcript language",
tries full-text summarization, falls back to chunking, and consolidates chunks.

Supports:
✅ Language identification (via langid) and enforcement in prompts
✅ Full transcript summarization if within token limit
✅ Chunking with line-based splitting and hard splits on long lines
✅ Summarization of chunks and consolidation into a final summary
✅ Handles .srt files (via srt package) and plain .txt transcripts
✅ Configurable word limits for chunk and final summaries



Install (in a venv):
    pip install .

Usage:
    summarize-interviews \
        --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --srt-file path/to/transcript.txt \
        --summary-words 500 \
        --summary-words-chunk 250 \
        --use-gpu yes \
        --device-id 0 \
        --cache-dir "/path/to/cache" \
        --hf-token <your_hf_token> \
        --output-dir output
"""
import argparse
import os
import time
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Optional deps
try:
    import srt
except ImportError:
    srt = None

try:
    import langid
except ImportError:
    langid = None


# ----------------------------
# Language detection
# ----------------------------

LANG_MAP = {
    "en": "English", "nl": "Dutch", "de": "German", "es": "Spanish",
    "fr": "French", "it": "Italian", "pt": "Portuguese",
    "pl": "Polish", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "el": "Greek", "tr": "Turkish", "ru": "Russian", "uk": "Ukrainian",
    "hi": "Hindi"
}

def detect_language_name(text: str, fallback: str = "the same language as the transcript") -> str:
    if not text:
        return fallback
    if langid is None:
        return fallback
    code, _ = langid.classify(text[:4000])
    return LANG_MAP.get(code, fallback)


# ----------------------------
# I/O
# ----------------------------

def load_srt_text(srt_path: str) -> str:
    if srt is None:
        raise RuntimeError("The 'srt' package is required to read .srt files. Install via: pip install srt")
    with open(srt_path, "r", encoding="utf-8") as f:
        segments = list(srt.parse(f.read()))
    return "\n".join(seg.content.replace("\n", " ") for seg in segments)

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------
# Chunking
# ----------------------------

def split_into_chunks(text: str, tokenizer, max_tokens: int) -> list:
    """
    Line-based chunking that respects a max token budget per chunk.
    Hard-splits overly long lines by words.
    """
    chunks, current_lines, current_tokens = [], [], 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line_tokens = len(tokenizer.tokenize(line))
        if line_tokens > max_tokens:
            for w in line.split():
                wt = len(tokenizer.tokenize(w))
                if current_tokens + wt > max_tokens:
                    chunks.append("\n".join(current_lines).strip())
                    current_lines, current_tokens = [], 0
                current_lines.append(w)
                current_tokens += wt
            continue

        if current_tokens + line_tokens > max_tokens:
            chunks.append("\n".join(current_lines).strip())
            current_lines, current_tokens = [line], line_tokens
        else:
            current_lines.append(line)
            current_tokens += line_tokens

    if current_lines:
        chunks.append("\n".join(current_lines).strip())

    return [c for c in chunks if c]


# ----------------------------
# Decoding helpers
# ----------------------------

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", flags=re.S | re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.S | re.I)
_DANGLING_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.I)

def count_words(text: str) -> int:
    """Count words in text, ignoring punctuation and extra whitespace."""
    return len(re.findall(r'\b\w+\b', text))

def trim_to_word_limit(text: str, word_limit: int) -> str:
    """Trim text to fit within word_limit, preserving complete sentences."""
    words = re.findall(r'\b\w+\b', text)
    if len(words) <= word_limit:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    trimmed = []
    word_count = 0
    for sentence in sentences:
        sentence_words = len(re.findall(r'\b\w+\b', sentence))
        if word_count + sentence_words <= word_limit:
            trimmed.append(sentence)
            word_count += sentence_words
        else:
            break
    return ' '.join(trimmed).strip()

def extract_summary_from_text(text: str, word_limit: int) -> str:
    """
    Remove everything before the last </think> (including the tag), all <think>...</think> blocks,
    and <summary> tags, keeping only the summary content. Trim to word_limit.
    """
    if not text:
        return ""
    # Normalize line endings
    text = text.replace("\r", "")
    
    # Remove everything before and including the last </think>
    matches = list(_DANGLING_THINK_CLOSE_RE.finditer(text))
    if matches:
        text = text[matches[-1].end():]
    
    # Remove any remaining <think>...</think> blocks
    text = _THINK_RE.sub("", text).strip()
    
    # Extract content between <summary>...</summary>, or use full text if absent
    m = _SUMMARY_RE.search(text)
    summary = m.group(1).strip() if m else text
    
    # Trim to word limit while preserving sentences
    summary = trim_to_word_limit(summary, word_limit)
    return summary


# ----------------------------
# Prompted summarization
# ----------------------------

def build_messages(language_name: str, text: str, word_limit: int, mode: str = "summarize") -> list:
    """
    mode: "summarize" (for transcript) or "consolidate" (merge chunk summaries)
    """
    sys_msg = (
        "You are a concise interview summarization assistant."
    )
    task_line = (
        "Summarize the transcript."
        if mode == "summarize"
        else "Consolidate these chunk summaries into a single coherent summary; remove redundancy."
    )
    user_msg = (
        f"LANGUAGE: {language_name}\n\n"
        f"{task_line}\n\n"
        f"TEXT:\n{text}\n\n"
        "INSTRUCTIONS:\n"
        "- If you need to reason or plan, do it step-by-step inside <think>...</think> tags.\n"
        "- Output ONLY a bold title (using **text**) followed by one or two paragraphs, with NO <summary> or <think> tags.\n"
        "- Write the summary ONLY in {language_name}.\n"
        "- Keep the summary content (title and paragraphs) under {word_limit} words.\n"
        "- Reasoning in <think> tags does not count toward the word limit.\n"
        "- Ensure the output is complete and not truncated."
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

def run_generation(model, tokenizer, messages, word_limit: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    # Estimate tokens needed: ~2 tokens per word + buffer for reasoning
    max_new_tokens = max(word_limit * 2 + 200, 1024)
    start = time.time()
    for attempt in range(2):  # Retry if output is truncated
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0][inputs.input_ids.shape[-1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        summary = extract_summary_from_text(raw, word_limit)
        # Check for truncation (no sentence-ending punctuation)
        if len(summary) > 0 and summary[-1] not in '.!?':
            print(f"Warning: Output may be truncated, retrying with increased tokens (attempt {attempt + 1})...")
            max_new_tokens += 200
            continue
        break
    print(f"Generation time: {time.time() - start:.2f}s")
    # Warn if summary is significantly under word limit
    word_count = count_words(summary)
    if word_count < word_limit * 0.7:
        print(f"Warning: Summary is {word_count} words, significantly below target {word_limit}")
    return summary

def summarize_transcript(model, tokenizer, language_name: str, text: str, word_limit: int) -> str:
    return run_generation(model, tokenizer, build_messages(language_name, text, word_limit, "summarize"), word_limit)

def consolidate_summaries(model, tokenizer, language_name: str, text: str, word_limit: int) -> str:
    return run_generation(model, tokenizer, build_messages(language_name, text, word_limit, "consolidate"), word_limit)


# ----------------------------
# Main pipeline
# ----------------------------

def summarize_interview(args):
    # Device selection (respect --device-id)
    use_gpu = (args.use_gpu.lower() == "yes") and torch.cuda.is_available()
    gpu_info = f"GPU available: {torch.cuda.is_available()} | Using GPU: {use_gpu}"
    print(gpu_info)

    # Quantization on GPU only
    bnb_config = None
    if use_gpu:
        supports_bf16 = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Resolve device_map based on --device-id visibility
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    multi = ("," in visible)
    device_map = "auto" if (use_gpu and multi) else ({"": 0} if use_gpu else None)

    print(f"CUDA_VISIBLE_DEVICES: {visible or '<not set>'}")
    print(f"device_map: {device_map}")

    # Load model/tokenizer
    print("Loading model/tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        dtype=torch.float16 if use_gpu else None,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    # Read input
    input_path = Path(args.srt_file)
    if input_path.suffix.lower() == ".srt":
        raw_text = load_srt_text(str(input_path))
    else:
        raw_text = load_txt(str(input_path))

    # Detect transcript language once (or override)
    language_name = args.force_language.strip() or detect_language_name(raw_text)
    print(f"Detected/forced language: {language_name}")

    # Token counts for decision
    raw_tokens = len(tokenizer.tokenize(raw_text))
    print(f"Raw transcript token count: {raw_tokens}")

    # Chunking parameters
    max_ctx_tokens = 3072
    reserved_for_generation = 512
    token_threshold = args.token_threshold

    chunks, chunk_summaries = [], []

    # Try full transcript path
    if raw_tokens <= token_threshold:
        print("Attempting full transcript summarization...")
        try:
            summary = summarize_transcript(model, tokenizer, language_name, raw_text, args.summary_words)
            chunk_summaries.append(summary)
            chunks = [raw_text]
        except Exception as e:
            print(f"Full summarization failed: {e}\nFalling back to chunking...")
            chunks = split_into_chunks(raw_text, tokenizer, max_ctx_tokens - reserved_for_generation)
    else:
        print(f"Transcript exceeds {token_threshold} tokens. Proceeding with chunking...")
        chunks = split_into_chunks(raw_text, tokenizer, max_ctx_tokens - reserved_for_generation)

    # Summarize chunks if needed
    if not chunk_summaries:
        print(f"Processing {len(chunks)} chunks...")
        for ch in tqdm(chunks, desc="Summarizing chunks"):
            s = summarize_transcript(model, tokenizer, language_name, ch, args.summary_words_chunk)
            chunk_summaries.append(s)

    # Consolidate if multiple chunks
    final_summary = "\n\n".join(chunk_summaries)
    if len(chunk_summaries) > 1:
        print("Consolidating chunk summaries...")
        final_summary = consolidate_summaries(model, tokenizer, language_name, final_summary, args.summary_words)

    # Final clean-up and word limit enforcement
    final_summary = extract_summary_from_text(final_summary, args.summary_words)
    word_count = count_words(final_summary)
    print(f"Final summary word count: {word_count}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}.summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"Summary saved to {out_path}")

    # Free memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleared model/tokenizer from memory.")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSeek Interview Summarizer (SRT/TXT)")
    parser.add_argument("--model-name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Hugging Face model ID")
    parser.add_argument("--srt-file", type=str, required=True,
                        help="Path to input .srt or speaker-labeled .txt transcript")
    parser.add_argument("--summary-words", type=int, default=500,
                        help="Target word count for the final (consolidated) summary")
    parser.add_argument("--summary-words-chunk", type=int, default=250,
                        help="Target word count per chunk summary (if chunking)")
    parser.add_argument("--token-threshold", type=int, default=4000,
                        help="If transcript tokens exceed this, use chunking")
    parser.add_argument("--use-gpu", type=str, default="yes", choices=["yes", "no"],
                        help="Use GPU if available (default: yes)")
    parser.add_argument("--device-id", type=str, default="",
                        help="CUDA device IDs, e.g., '0' or '0,1'. If set, we export CUDA_VISIBLE_DEVICES accordingly.")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to cache model weights (HF default if omitted)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to write output summaries")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face token (if needed for gated/private models)")
    parser.add_argument("--force-language", type=str, default="",
                        help="Override detected language with a human-readable name (e.g., 'Spanish', 'Dutch').")
    args = parser.parse_args()

    # Validate word limits
    if args.summary_words <= 0 or args.summary_words_chunk <= 0:
        raise ValueError("Word limits must be positive integers")

    # Respect --device-id BEFORE any CUDA init
    if args.use_gpu.lower() == "yes" and args.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        print(f"Set CUDA_VISIBLE_DEVICES={args.device_id}")

    summarize_interview(args)


if __name__ == "__main__":
    main()