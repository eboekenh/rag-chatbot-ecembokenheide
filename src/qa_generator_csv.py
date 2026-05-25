"""
qa_generator_csv.py – Generate Q&A pairs from cleaned api_functions_v2.csv.

Each row (one function) is treated as one chunk.
Writes to output CSV every WRITE_EVERY pairs so no data is lost on interruption.
Resumes automatically if interrupted — already-processed functions are skipped.
"""

import os
import sys
import json
import re
import time

import pandas as pd
from groq import Groq
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import QA_DATASET_PATH, GROQ_API_KEY, GROQ_QA_MODEL

CSV_INPUT   = "data/processed/api_functions_v2_cleaned.csv"
CSV_OUTPUT  = "data/processed/qa_dataset_v2.csv"
WRITE_EVERY = 3   # flush buffer to disk after every N pairs

# Fallback filter — skipped even if not manually removed from CSV
SKIP_SECTIONS = {"Date Offsets", "Extensions", "Testing"}

QA_PROMPT = """You are building a Q&A dataset for a RAG chatbot that helps data scientists, data analysts, and data engineers use the pandas library effectively. Your goal is to generate question-answer pairs that teach users HOW to use pandas in real data workflows — not just what a function is called.

Given the pandas documentation below, generate exactly {n} question-answer pairs.

RULES:
- Questions must be general and standalone — a user must be able to ask them WITHOUT having read the docs
- NEVER reference "the example", "the code", "in the example code", or example variable names (df, df2, s, s1, result, etc.)
- Questions must be practical and relevant to data work: purpose, how to use it, key parameters, return value, or common use cases
- Answers must be explanatory: describe what the function does, what the result looks like, and include a short code example when relevant
- If code examples are provided, use them to write a practical usage question (not about what the example variable contains)

Return ONLY a valid JSON object with a "pairs" key. No explanations, no markdown.

Format:
{{"pairs": [
  {{"question": "...", "answer": "..."}},
  ...
]}}

FUNCTION: {function_name}({parameters})

DESCRIPTION:
{description}"""


def generate_qa_for_row(
    row: dict,
    client: Groq,
    n_pairs: int = 3,
    max_retries: int = 2,
    max_rate_limit_retries: int = 4,
) -> list[dict]:
    """Call Groq and return a list of {question, answer, source_page} dicts."""
    description = str(row.get("description", "")).strip()
    if len(description) < 30:
        return []

    prompt = QA_PROMPT.format(
        n=n_pairs,
        function_name=row.get("function_name", ""),
        parameters=row.get("parameters", ""),
        description=description[:1500],  # cap to stay under TPM limit
    )

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

                # Strip markdown code fences
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

                # Extract first JSON object
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    raw = match.group(0)

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
                            "answer":   str(p["answer"]).strip(),
                            "source_page": str(row.get("source_url", "")),
                        })
                return valid

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries:
                    print(f"  [RETRY {attempt + 1}] JSON parse error: {e}")
                else:
                    print(f"  [SKIP] {row.get('function_name')}: failed after {max_retries + 1} attempts")
                    return []
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    break  # exit inner loop → sleep outer
                print(f"  [ERROR] {row.get('function_name')}: {e}")
                return []
        else:
            break  # no rate limit → done

        rate_limit_attempts += 1
        if rate_limit_attempts <= max_rate_limit_retries:
            wait = 60 + (rate_limit_attempts - 1) * 30  # 60s, 90s, 120s, 150s
            print(f"  [RATE LIMIT] Waiting {wait}s... ({rate_limit_attempts}/{max_rate_limit_retries})")
            time.sleep(wait)
        else:
            print(f"  [SKIP] Rate limit retries exhausted.")

    return []


def flush(buffer: list[dict], out_path: str) -> None:
    """Append buffer to CSV and clear it."""
    if not buffer:
        return
    df = pd.DataFrame(buffer, columns=["question", "answer", "source_page"])
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    df.to_csv(out_path, mode="a", index=False, header=write_header)
    buffer.clear()


def run_qa_generator_csv(csv_input: str = CSV_INPUT, out_path: str = None) -> None:
    if out_path is None:
        out_path = CSV_OUTPUT

    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")

    # Load input CSV
    df = pd.read_csv(csv_input)
    print(f"Loaded {len(df)} rows from {csv_input}")

    # Filter out irrelevant sections (fallback — user may have already cleaned)
    before = len(df)
    df = df[~df["section"].isin(SKIP_SECTIONS)]
    skipped = before - len(df)
    if skipped:
        print(f"Filtered {skipped} rows from: {SKIP_SECTIONS}")

    # Resume: skip already-processed source_urls
    already_done: set[str] = set()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    if os.path.exists(out_path):
        try:
            existing = pd.read_csv(out_path)
            already_done = set(existing["source_page"].dropna().unique())
            print(f"Resuming: {len(already_done)} source URL(s) already processed.")
        except Exception:
            pass

    remaining = df[~df["source_url"].isin(already_done)]
    print(f"Functions to process: {len(remaining)} / {len(df)}\n")

    client = Groq(api_key=GROQ_API_KEY)
    buffer: list[dict] = []
    total_pairs = 0

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Generating Q&A"):
        pairs = generate_qa_for_row(row.to_dict(), client, n_pairs=3)
        buffer.extend(pairs)
        total_pairs += len(pairs)

        # Write every WRITE_EVERY pairs
        if len(buffer) >= WRITE_EVERY:
            flush(buffer, out_path)
            tqdm.write(f"  Saved {total_pairs} pairs so far → {out_path}")

        time.sleep(12)  # ~5 calls/min → ~7500 tokens/min, safely under 14400 TPM free tier

    # Flush remaining buffer
    flush(buffer, out_path)

    # Final deduplication
    if os.path.exists(out_path):
        final_df = pd.read_csv(out_path)
        before = len(final_df)
        final_df = final_df.drop_duplicates(subset=["question"])
        final_df.to_csv(out_path, index=False)
        print(f"\nDone. {len(final_df)} unique pairs (removed {before - len(final_df)} duplicates) → {out_path}")


if __name__ == "__main__":
    run_qa_generator_csv()
