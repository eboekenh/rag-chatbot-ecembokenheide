import os
import sys
import json
import re
import time

import pandas as pd
from groq import Groq
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RAW_DATA_DIR,
    QA_DATASET_PATH,
    GROQ_API_KEY,
    GROQ_QA_MODEL,
)

QA_PROMPT = """You are a Q&A dataset generator for a RAG chatbot about the pandas library.

Given the text below, generate exactly {n} question-answer pairs that test real comprehension.
- Questions must be specific and answerable from the text only
- Answers must be concise but complete (2-4 sentences)
- Cover different aspects of the text (don't repeat similar questions)

Return ONLY a valid JSON object with a "pairs" key containing an array. No explanations, no markdown, no preamble.

Format:
{{"pairs": [
  {{"question": "...", "answer": "..."}},
  ...
]}}

TEXT:
{chunk}"""


def load_raw_pages(raw_dir: str) -> list[dict]:
    pages = []
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(raw_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        # Parse header metadata
        url = ""
        title = filename
        content_start = 0
        for i, line in enumerate(raw.splitlines()):
            if line.startswith("URL: "):
                url = line[5:].strip()
            elif line.startswith("TITLE: "):
                title = line[7:].strip()
            elif line.startswith("-" * 10):
                content_start = i + 1
                break

        content = "\n".join(raw.splitlines()[content_start:]).strip()
        if content:
            pages.append({"filename": filename, "url": url, "title": title, "content": content})

    print(f"Loaded {len(pages)} pages from {raw_dir}")
    return pages


def chunk_for_qa(content: str, max_chars: int = 3000) -> list[str]:
    """Split content into chunks that respect section boundaries.

    Sections are delimited by lines starting with '## ' (added by scraper for
    h2-h6 headings). When a section is too long to fit in one chunk, each
    continuation chunk gets a '[continued: <heading>]' prefix so the LLM
    always has context about which section it is reading.
    """
    # Split into sections at heading markers
    raw_sections = re.split(r"(?=\n## )", content)
    sections = [s.strip() for s in raw_sections if s.strip()]

    chunks: list[str] = []
    current = ""

    for section in sections:
        # Extract heading line for use as context prefix in continuations
        first_line = section.split("\n")[0].strip()
        heading = first_line[3:] if first_line.startswith("## ") else first_line

        if len(section) <= max_chars:
            # Section fits — try to append to current chunk
            if len(current) + len(section) + 2 <= max_chars:
                current += ("\n\n" if current else "") + section
            else:
                if current:
                    chunks.append(current.strip())
                current = section
        else:
            # Section too long — flush current and split by paragraphs
            if current:
                chunks.append(current.strip())
                current = ""
            paragraphs = section.split("\n\n")
            sub = ""
            is_first = True
            for para in paragraphs:
                if len(sub) + len(para) + 2 <= max_chars:
                    sub += ("\n\n" if sub else "") + para
                else:
                    if sub:
                        chunks.append(sub.strip())
                    if len(para) > max_chars:
                        # Single paragraph too long: split by sentence
                        sentences = re.split(r"(?<=[.!?])\s+", para)
                        sub = "" if is_first else f"[continued: {heading}]\n"
                        for sent in sentences:
                            if len(sub) + len(sent) + 1 <= max_chars:
                                sub += (" " if sub else "") + sent
                            else:
                                if sub:
                                    chunks.append(sub.strip())
                                sub = f"[continued: {heading}]\n{sent}"
                    else:
                        # Start new sub-chunk; add heading context if not first
                        sub = para if is_first else f"[continued: {heading}]\n\n{para}"
                is_first = False
            if sub:
                chunks.append(sub.strip())

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 100]


def generate_qa_for_chunk(
    chunk: str,
    source_url: str,
    client: Groq,
    n_pairs: int = 5,
    max_retries: int = 2,
    max_rate_limit_retries: int = 4,
) -> list[dict]:
    """Call Groq to generate n_pairs Q&A pairs from chunk.

    JSON parse errors: retried up to max_retries times (fast).
    Rate limit errors: retried up to max_rate_limit_retries times (65s sleep each).
    """
    prompt = QA_PROMPT.format(n=n_pairs, chunk=chunk)
    rate_limit_attempts = 0

    while rate_limit_attempts <= max_rate_limit_retries:
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=GROQ_QA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                )
                raw = response.choices[0].message.content.strip()

                # Strip markdown code fences if present
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

                # Extract first JSON object if extra text is present
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    raw = match.group(0)

                # Fix common JSON issues: unescaped single quotes inside strings
                raw = raw.replace("\\'", "'")

                parsed = json.loads(raw)
                pairs = parsed.get("pairs", parsed) if isinstance(parsed, dict) else parsed
                if not isinstance(pairs, list):
                    raise ValueError("Response does not contain a pairs array")

                valid = []
                for p in pairs:
                    if isinstance(p, dict) and "question" in p and "answer" in p:
                        valid.append({
                            "question": str(p["question"]).strip(),
                            "answer": str(p["answer"]).strip(),
                            "source_page": source_url,
                        })
                return valid

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries:
                    print(f"  [RETRY {attempt + 1}] JSON parse error: {e}")
                else:
                    print(f"  [SKIP] Failed after {max_retries + 1} attempts: {e}")
                    return []
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    break  # exit inner loop → sleep and retry outer
                print(f"  [ERROR] Groq request failed: {e}")
                return []
        else:
            break  # inner loop finished without rate limit → done

        # Rate limit hit: sleep and retry outer loop
        rate_limit_attempts += 1
        if rate_limit_attempts <= max_rate_limit_retries:
            wait = 60
            print(f"  [RATE LIMIT] Waiting {wait}s... ({rate_limit_attempts}/{max_rate_limit_retries})")
            time.sleep(wait)
        else:
            print(f"  [SKIP] Rate limit retries exhausted for this chunk.")

    return []


def run_qa_generator(raw_dir: str = None, out_path: str = None) -> None:
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR
    if out_path is None:
        out_path = QA_DATASET_PATH

    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")

    pages = load_raw_pages(raw_dir)

    if not pages:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}. Run scraper.py first.")

    # Resume: find which pages are already done
    already_done: set[str] = set()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    if os.path.exists(out_path):
        try:
            existing_df = pd.read_csv(out_path)
            already_done = set(existing_df["source_page"].dropna().unique())
            print(f"Resuming: {len(already_done)} page(s) already processed, skipping them.")
        except Exception:
            pass

    remaining = [p for p in pages if p["url"] not in already_done]
    print(f"Pages to process: {len(remaining)} / {len(pages)}")

    # Single client shared across all chunks (avoids creating ~1500 instances)
    client = Groq(api_key=GROQ_API_KEY)

    for page in tqdm(remaining, desc="Generating Q&A"):
        chunks = chunk_for_qa(page["content"], max_chars=1500)
        page_pairs = []

        for chunk in tqdm(chunks, desc=f"  Chunks [{page['title'][:30]}]", leave=False):
            # Scale n_pairs by chunk size: 1 pair per ~300 chars, min 2 max 5
            n_pairs = max(2, min(5, len(chunk) // 300))
            pairs = generate_qa_for_chunk(
                chunk=chunk,
                source_url=page["url"],
                client=client,
                n_pairs=n_pairs,
            )
            page_pairs.extend(pairs)
            time.sleep(4)  # proactive delay to stay under Groq rate limits

        print(f"  {page['title']}: {len(chunks)} chunks → {len(page_pairs)} pairs")

        if page_pairs:
            page_df = pd.DataFrame(page_pairs, columns=["question", "answer", "source_page"])
            write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
            page_df.to_csv(out_path, mode="a", index=False, header=write_header)
            print(f"  Saved {len(page_pairs)} pairs for '{page['title']}' → {out_path}")

    # Final deduplication pass
    if os.path.exists(out_path):
        final_df = pd.read_csv(out_path)
        before = len(final_df)
        final_df = final_df.drop_duplicates(subset=["question"])
        final_df.to_csv(out_path, index=False)
        print(f"\nDone. {len(final_df)} unique pairs total (removed {before - len(final_df)} duplicates).")


if __name__ == "__main__":
    run_qa_generator()