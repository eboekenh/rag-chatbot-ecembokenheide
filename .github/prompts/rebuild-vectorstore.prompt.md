---
description: "Rebuild the RAG vector store from scratch: scrape docs, generate Q&A pairs, and populate ChromaDB."
name: "Rebuild Vector Store"
argument-hint: "Generator to use: groq (default) | ollama | csv"
agent: "agent"
tools: ["run_in_terminal", "read_file", "create_file"]
---

Rebuild the full RAG pipeline for this pandas documentation chatbot using the **$input** generator (default: `groq`).

## Steps

1. **Delete the old vector store** so `vector_store.py` doesn't raise on existing collections:
   ```powershell
   Remove-Item -Recurse -Force chroma_db\docs, chroma_db\qa -ErrorAction SilentlyContinue
   ```

2. **Phase 1 — Scrape docs** (skip if `data/raw/` already has 23 `.txt` files):
   ```powershell
   python src/scraper.py
   ```

3. **Phase 2 — Generate Q&A pairs** using the chosen generator:
   - `groq` (default): `python src/qa_generator.py` — requires `GROQ_API_KEY` in `.env`
   - `ollama`: `python src/qa_generator_ollama.py` — requires `ollama serve` running on port 11434
   - `csv`: `python src/qa_generator_csv.py` — reads from `data/api_functions_v2.csv`

4. **Phase 3 — Build vector databases**:
   ```powershell
   python src/vector_store.py
   ```

5. **Verify** both collections exist under `chroma_db/docs/` and `chroma_db/qa/`.

## Notes

- All constants (paths, thresholds, model names) are in [`config.py`](../../config.py) — never hardcode them.
- Q&A generators are resumable: if interrupted, re-running skips already-processed `source_page` entries.  
  Delete `data/processed/qa_dataset.csv` only if you want a full regeneration.
- Groq rate limiting: 429 errors trigger 60–150s automatic backoff — do not reduce sleep intervals.
