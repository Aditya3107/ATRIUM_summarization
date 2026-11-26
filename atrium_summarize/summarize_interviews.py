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
✅ Server-mode for use by [sumservice](https://github.com/proycon/sumservice)


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
import sys
import argparse
import os
import time
import re
import socket
import threading
import traceback
import signal
from typing import Optional
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
        f"INSTRUCTIONS:\n"
        f"- If you need to reason or plan, do it step-by-step inside <think>...</think> tags.\n"
        f"- Output ONLY a bold title (using **text**) followed by one or two paragraphs, with NO <summary> or <think> tags.\n"
        f"- Write the summary ONLY in {language_name}.\n"
        f"- Keep the summary content (title and paragraphs) under {word_limit} words.\n"
        f"- Reasoning in <think> tags does not count toward the word limit.\n"
        f"- Ensure the output is complete and not truncated."
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
    summary = ""
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

class SumService:

    def __init__(self, args):
        use_gpu = (args.use_gpu.lower() == "yes") and torch.cuda.is_available()
        gpu_info = f"GPU available: {torch.cuda.is_available()} | Using GPU: {use_gpu}"
        print(gpu_info, file=sys.stderr)

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

        print(f"CUDA_VISIBLE_DEVICES: {visible or '<not set>'}", file=sys.stderr)
        print(f"device_map: {device_map}", file=sys.stderr)

        # Load model/tokenizer
        print("Loading model/tokenizer...", file=sys.stderr)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=device_map,
            quantization_config=bnb_config,
            dtype=torch.float16 if use_gpu else None,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        self.model.config.use_cache = True
        self.model.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )


    def summarize_interview(self, input_file: str, output_dir: str, force_language: Optional[str], token_threshold: int, summary_words: int, summary_words_chunk: int) -> bool:
        # Read input
        input_path = Path(input_file)
        if input_path.suffix.lower() == ".srt":
            raw_text = load_srt_text(str(input_path))
        else:
            raw_text = load_txt(str(input_path))

        # Detect transcript language once (or override)
        language_name = force_language if force_language else detect_language_name(raw_text)
        print(f"Detected/forced language: {language_name}")

        # Token counts for decision
        raw_tokens = len(self.tokenizer.tokenize(raw_text))
        print(f"Raw transcript token count: {raw_tokens}")

        # Chunking parameters
        max_ctx_tokens = 3072
        reserved_for_generation = 512
        token_threshold = token_threshold

        chunks, chunk_summaries = [], []

        # Try full transcript path
        if raw_tokens <= token_threshold:
            print("Attempting full transcript summarization...", file=sys.stderr)
            try:
                summary = summarize_transcript(self.model, self.tokenizer, language_name, raw_text, summary_words)
                chunk_summaries.append(summary)
                chunks = [raw_text]
            except Exception as e:
                print(f"Full summarization failed: {e}\nFalling back to chunking...", file=sys.stderr)
                chunks = split_into_chunks(raw_text, self.tokenizer, max_ctx_tokens - reserved_for_generation)
        else:
            print(f"Transcript exceeds {token_threshold} tokens. Proceeding with chunking...", file=sys.stderr)
            chunks = split_into_chunks(raw_text, self.tokenizer, max_ctx_tokens - reserved_for_generation)

        # Summarize chunks if needed
        if not chunk_summaries:
            print(f"Processing {len(chunks)} chunks...", file=sys.stderr)
            for ch in tqdm(chunks, desc="Summarizing chunks"):
                s = summarize_transcript(self.model, self.tokenizer, language_name, ch, summary_words_chunk)
                chunk_summaries.append(s)

        # Consolidate if multiple chunks
        final_summary = "\n\n".join(chunk_summaries)
        if len(chunk_summaries) > 1:
            print("Consolidating chunk summaries...", file=sys.stderr)
            final_summary = consolidate_summaries(self.model, self.tokenizer, language_name, final_summary, summary_words)

        # Final clean-up and word limit enforcement
        final_summary = extract_summary_from_text(final_summary, summary_words)
        word_count = count_words(final_summary)
        print(f"Final summary word count: {word_count}", file=sys.stderr)

        # Save
        output_dir_path: Path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = output_dir_path / f"{input_path.stem}.summary.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_summary)

        print(f"Summary saved to {out_path}", file=sys.stderr)
        return True


# ----------------------------
# Server
# ----------------------------

def shutdown(signalnum, _):
    global terminate
    print(f"Shutting down, received signal {signalnum}", file=sys.stderr)
    terminate.set()

def client_handler(conn):
    """Handle incoming client connections"""
    conn.settimeout(2)
    try:
        with conn:
            while not terminate.is_set():
                data = conn.recv(1024)
                if not data:
                    break
                try:
                    filename, output_dir, force_language,token_threshold, summary_words, summary_words_chunk = str(data,'utf-8').strip().split("\t")
                except:
                    conn.sendall(b"ERROR: Expected tab-separated fields: filename, output_dir, force_language, token_threshold, summary_words, summary_words_chunk\n")
                    continue
                try:
                    conn.sendall(b"Processing " + os.path.basename(filename).encode('utf-8') + b" ...\n")
                    if force_language in ("","-","null","False","None"):
                        force_language = None
                    if service.summarize_interview(filename, output_dir, force_language, int(token_threshold), int(summary_words), int(summary_words_chunk)):
                        conn.sendall(b"Finished " + os.path.basename(filename).encode('utf-8') + b"\n")
                    else:
                        conn.sendall(b"ERROR: Invalid file or file not found\n")
                except Exception as e:
                    traceback.print_exc()
                    conn.sendall(f"ERROR: {e}".encode('utf-8') + b"\n")
                #important, send EOF and close connection, client may only send one instruction per connection
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
                return
    except OSError as e:
        print(f"Client connection ended: {e}", file=sys.stderr)

# ----------------------------
# CLI
# ----------------------------

def main():
    global service, terminate

    parser = argparse.ArgumentParser(description="DeepSeek Interview Summarizer (SRT/TXT)")
    parser.add_argument("--model-name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Hugging Face model ID")
    parser.add_argument("--srt-file", type=str,
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
    parser.add_argument("--server", type=int,
                        help="Run as a server on the selected port")
    args = parser.parse_args()

    # Validate word limits
    if args.summary_words <= 0 or args.summary_words_chunk <= 0:
        raise ValueError("Word limits must be positive integers")

    # Respect --device-id BEFORE any CUDA init
    if args.use_gpu.lower() == "yes" and args.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        print(f"Set CUDA_VISIBLE_DEVICES={args.device_id}")

    # Catch signals, we must be able to terminate gracefully on request
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    service = SumService(args)

    if args.srt_file:
        service.summarize_interview(args.srt_file, args.output_dir, args.force_language, args.token_threshold, args.summary_words, args.summary_words_chunk)
    elif args.server:
        port= int(args.server)
        print("READY",file=sys.stderr);

        terminate = threading.Event()

        # Start the server, allow multiple clients
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", int(port)))
            server.listen()
            server.settimeout(0.5) #timeout needed so we can catch signals and shutdown if needed

            threads = []

            while not terminate.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                thread = threading.Thread(target=client_handler, args=(conn,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join(timeout=4)
    else:
        raise ValueError("Provide --srt-file or set --server")


if __name__ == "__main__":
    main()
