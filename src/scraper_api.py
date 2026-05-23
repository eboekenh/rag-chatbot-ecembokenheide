"""
scraper_api.py – Scrapes the pandas API reference and saves each function as a CSV row.

Output: data/api_functions.csv
Columns: section, subsection, function_name, parameters, description, source_url

description contains: brief summary + full docstring + code examples (all in one field).
Each row = one function/attribute → perfect structured chunk for QA generation.
"""

import os
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCRAPE_DELAY

BASE_URL = "https://pandas.pydata.org/docs/reference/"
CSV_OUTPUT = "data/api_functions_v2.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; educational-research-bot/1.0; "
        "RAG chatbot assignment)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# API reference pages to scrape (section_name, relative_url)
API_PAGES = [
    ("DataFrame",        "frame.html"),
    ("Series",           "series.html"),
    ("Index",            "indexing.html"),
    ("Arrays",           "arrays.html"),
    # ("Date Offsets",   "offset_frequency.html"),  # 688 entries, very repetitive
    ("Input/Output",     "io.html"),
    ("General Functions","general_functions.html"),
    ("GroupBy",          "groupby.html"),
    ("Resampling",       "resampling.html"),
    ("Window",           "window.html"),
    ("Plotting",         "plotting.html"),
    ("Styling",          "style.html"),
    # ("Extensions",     "extensions.html"),  # for library developers only
    # ("Testing",        "testing.html"),      # for pandas contributors only
]


def fetch_page(url: str) -> BeautifulSoup | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except requests.exceptions.HTTPError as e:
        print(f"  [SKIP] HTTP {e.response.status_code} – {url}")
    except requests.exceptions.RequestException as e:
        print(f"  [SKIP] Request failed – {url}: {e}")
    return None


def fetch_detail(url: str) -> str:
    """Visit individual function page and return full description + examples as one text block.

    Format:
        {full description paragraphs}

        Example:
        {code block 1}

        {code block 2}
    """
    soup = fetch_page(url)
    if soup is None:
        return ""

    parts = []

    # Full description: all <p> tags directly inside the main <dd>
    main_dd = soup.select_one("dl.py > dd")
    if main_dd:
        for p in main_dd.find_all("p", recursive=False):
            text = p.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)

    # Examples section — Sphinx marks it with <p class="rubric">Examples</p>
    examples_rubric = soup.find("p", class_="rubric", string=lambda t: t and "Examples" in t)
    if examples_rubric:
        code_blocks = []
        for sibling in examples_rubric.find_next_siblings():
            # Stop at next rubric (e.g. "Notes", "See Also")
            if sibling.name == "p" and "rubric" in sibling.get("class", []):
                break
            code_blocks.extend(sibling.find_all("pre"))
        if code_blocks:
            parts.append("Example:")
            for pre in code_blocks[:3]:  # max 3 code blocks
                code = pre.get_text(strip=True)
                lines = [
                    re.sub(r"^>>> ?", "", line)
                    for line in code.splitlines()
                    if line.strip() and not line.strip().startswith("...")
                ]
                if lines:
                    parts.append("\n".join(lines))

    return "\n\n".join(parts).strip()


def parse_func_cell(td) -> tuple[str, str]:
    """Extract (function_name, parameters) from a table cell like
    'DataFrame.info([verbose, buf, max_cols, ...])'.
    """
    raw = td.get_text(separator="", strip=True)
    raw = raw.replace("\xa0", " ")
    match = re.match(r"^([^\(]+)(\(.*\))?$", raw)
    if match:
        name = match.group(1).strip()
        params = match.group(2) or ""
        params = params.strip("()").strip()
        return name, params
    return raw, ""


def scrape_api_page(section: str, rel_url: str) -> list[dict]:
    """Scrape one API reference page and return list of function dicts."""
    url = BASE_URL + rel_url
    soup = fetch_page(url)
    if soup is None:
        return []

    rows = []
    tables = soup.select("table.autosummary")

    for table in tables:
        heading = table.find_previous(["h2", "h3"])
        subsection = heading.get_text(strip=True).rstrip("#").strip() if heading else section

        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            func_name, parameters = parse_func_cell(tds[0])
            brief = tds[1].get_text(separator=" ", strip=True)

            a_tag = tds[0].find("a", href=True)
            if a_tag:
                href = a_tag["href"].split("#")[0]
                source_url = BASE_URL + href
            else:
                source_url = url

            if func_name:
                rows.append({
                    "section": section,
                    "subsection": subsection,
                    "function_name": func_name,
                    "parameters": parameters,
                    "brief": brief,
                    "source_url": source_url,
                })

    return rows


def run_scraper_api(out_path: str = CSV_OUTPUT) -> str:
    # Step 1: collect all function rows from overview pages
    all_rows = []
    failed = []

    print(f"Step 1: Scraping {len(API_PAGES)} overview pages → collecting function list")
    for section, rel_url in tqdm(API_PAGES, desc="Overview pages"):
        rows = scrape_api_page(section, rel_url)
        if rows:
            all_rows.extend(rows)
            print(f"  {section}: {len(rows)} functions")
        else:
            failed.append(section)
        time.sleep(SCRAPE_DELAY)

    print(f"\nStep 2: Fetching details + examples for {len(all_rows)} functions ...")
    for row in tqdm(all_rows, desc="Detail pages"):
        detail = fetch_detail(row["source_url"])
        # Combine: detail (full desc + examples) if available, else fall back to brief
        row["description"] = detail if detail else row["brief"]
        time.sleep(1)  # polite delay per function page

    df = pd.DataFrame(
        all_rows,
        columns=["section", "subsection", "function_name", "parameters", "description", "source_url"],
    )
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nFertig: {len(df)} Funktionen aus {len(API_PAGES) - len(failed)} Seiten → {out_path}")
    if failed:
        print("Fehlgeschlagen:", failed)

    return out_path


if __name__ == "__main__":
    path = run_scraper_api()
    df = pd.read_csv(path)
    print(f"\nVorschau (erste 3 Funktionen mit Beispielen):")
    for _, row in df[df["description"].str.contains("Example:", na=False)].head(3).iterrows():
        print(f"\n  [{row['section']}] {row['function_name']}")
        print("  " + row["description"][:300].replace("\n", "\n  "))
    with_examples = df["description"].str.contains("Example:", na=False).sum()
    print(f"\nFunktionen mit Beispielen: {with_examples} / {len(df)}")
    print(f"\nSektionen:")
    print(df.groupby("section").size().to_string())
