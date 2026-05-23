"""
scraper_csv.py – Scrapes pandas docs and saves each section as a CSV row.

Output: data/sections.csv
Columns: url, page_title, section, content, scraped_at

Each row = one section (split at ## headings) → natural chunk for QA generator.
"""

import os
import re
import sys
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCRAPE_TARGETS, SCRAPE_DELAY
from src.scraper import fetch_page, clean_html

CSV_OUTPUT = "data/sections.csv"


def split_into_sections(page_data: dict) -> list[dict]:
    """Split cleaned page text into sections at ## heading markers.

    Returns a list of dicts with keys:
        url, page_title, section, content, scraped_at
    """
    lines = page_data["content"].splitlines()
    sections = []
    current_heading = "Introduction"
    current_lines = []

    for line in lines:
        if line.startswith("## "):
            # Save accumulated content under previous heading
            text = "\n".join(current_lines).strip()
            if text:
                sections.append({
                    "url": page_data["url"],
                    "page_title": page_data["title"],
                    "section": current_heading,
                    "content": text,
                    "scraped_at": page_data["scraped_at"],
                })
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    text = "\n".join(current_lines).strip()
    if text:
        sections.append({
            "url": page_data["url"],
            "page_title": page_data["title"],
            "section": current_heading,
            "content": text,
            "scraped_at": page_data["scraped_at"],
        })

    return sections


def run_scraper_csv(
    targets: list[str] | None = None,
    out_path: str = CSV_OUTPUT,
) -> str:
    if targets is None:
        targets = SCRAPE_TARGETS

    all_sections = []
    failed = []

    print(f"Scraping {len(targets)} pages → {out_path}")
    for url in tqdm(targets, desc="Scraping"):
        soup = fetch_page(url)
        if soup is None:
            failed.append(url)
            time.sleep(SCRAPE_DELAY)
            continue

        page_data = clean_html(soup, url)

        if len(page_data["content"]) < 100:
            print(f"  [WARN] Very short content ({len(page_data['content'])} chars) – {url}")

        sections = split_into_sections(page_data)
        all_sections.extend(sections)
        print(f"  {page_data['title']}: {len(sections)} sections")
        time.sleep(SCRAPE_DELAY)

    df = pd.DataFrame(all_sections, columns=["url", "page_title", "section", "content", "scraped_at"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nDone: {len(df)} sections from {len(targets) - len(failed)} pages → {out_path}")
    if failed:
        print("Failed URLs:")
        for u in failed:
            print(f"  - {u}")

    return out_path


if __name__ == "__main__":
    path = run_scraper_csv()
    df = pd.read_csv(path)
    print(f"\nVorschau (erste 3 Zeilen):")
    for _, row in df.head(3).iterrows():
        print(f"\n  [{row['page_title']}] > {row['section']}")
        print(f"  {row['content'][:200]}...")
    print(f"\nGesamte Sektionen: {len(df)}")
    print(f"Spalten: {list(df.columns)}")
