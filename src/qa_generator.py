import os
import sys
import json
import time
import re

import pandas as pd
from tqdm import tqdm
from groq import Groq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GROQ_API_KEY,
    RAW_DATA_DIR,
    QA_DATASET_PATH,
    LLM_MODEL,
)

QA_PROMPT = """You are a Q&A dataset generator for a RAG chatbot about the pandas library.

Given the text below, generate exactly {n} question-answer pairs that test real comprehension.
- Questions must be specific and answerable from the text only
- Answers must be concise but complete (2-4 sentences)
- Cover different aspects of the text (don't repeat similar questions)

Return ONLY a valid JSON array. No explanations, no markdown, no preamble.

Format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

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
    if len(content) <= max_chars:
        return [content]

    chunks = []
    paragraphs = content.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current.strip())
            # If single paragraph exceeds max_chars, split by sentences
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current += (" " if current else "") + sent
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sent
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 100]


def generate_qa_for_chunk(
    chunk: str,
    source_url: str,
    client: Groq,
    n_pairs: int = 8,
    max_retries: int = 2,
) -> list[dict]:
    prompt = QA_PROMPT.format(n=n_pairs, chunk=chunk)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                raise ValueError("Response is not a JSON array")

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
                time.sleep(2)
            else:
                print(f"  [SKIP] Failed after {max_retries + 1} attempts: {e}")

    return []


def deduplicate(pairs: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for p in pairs:
        key = p["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def run_qa_generator(raw_dir: str = None, out_path: str = None) -> None:
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR
    if out_path is None:
        out_path = QA_DATASET_PATH

    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY not set. Create a .env file with your key.")

    client = Groq(api_key=GROQ_API_KEY)
    pages = load_raw_pages(raw_dir)

    if not pages:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}. Run scraper.py first.")

    all_pairs = []

    for page in tqdm(pages, desc="Generating Q&A"):
        chunks = chunk_for_qa(page["content"], max_chars=3000)
        page_pairs = []

        for chunk in chunks:
            pairs = generate_qa_for_chunk(
                chunk=chunk,
                source_url=page["url"],
                client=client,
                n_pairs=8,
            )
            page_pairs.extend(pairs)
            # Respect Groq rate limits
            time.sleep(1.5)

        print(f"  {page['title']}: {len(chunks)} chunks → {len(page_pairs)} pairs")
        all_pairs.extend(page_pairs)

    all_pairs = deduplicate(all_pairs)
    print(f"\nTotal Q&A pairs after deduplication: {len(all_pairs)}")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df = pd.DataFrame(all_pairs, columns=["question", "answer", "source_page"])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} pairs to {out_path}")


if __name__ == "__main__":
    run_qa_generator()
