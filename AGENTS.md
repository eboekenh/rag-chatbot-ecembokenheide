# RAG QA Chatbot — Agent Instructions

Pandas documentation RAG chatbot. Answers pandas questions via hybrid retrieval (Q&A vector store + doc chunks). Built with ChromaDB, Sentence Transformers, Groq LLM, and Streamlit.

## Environment Setup

**Python 3.10+ required.** A `.venv` is present in the project root.

```powershell
# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Or on bash/cmd
source .venv/Scripts/activate
```

**Required `.env` file** (project root, not committed):
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at https://console.groq.com.

**Optional — for Ollama-based generators:**
```powershell
ollama serve          # must be running on port 11434
ollama pull llama3.2:3b
```

## Build & Run Commands

```powershell
# Full pipeline (run in order on a fresh setup)
python src/scraper.py            # Phase 1: scrape 23 pandas doc pages → data/raw/
python src/qa_generator.py       # Phase 2: generate Q&A via Groq → data/processed/qa_dataset.csv
python src/vector_store.py       # Phase 3: build chroma_db/docs/ and chroma_db/qa/
streamlit run app.py             # Phase 5: start chat UI at http://localhost:8501

# Alternative QA generators (Phase 2 variants)
python src/qa_generator_ollama.py  # requires ollama serve + llama3.2:3b
python src/qa_generator_csv.py     # generates from data/api_functions_v2.csv

# Tests
python test_ollama_chatbot.py    # compare Ollama vs Groq Q&A datasets
python test_ollama.py            # verify Ollama connectivity (port 11434)
```

## Architecture

**5-phase pipeline:**

| Phase | Script | Input → Output |
|-------|--------|----------------|
| 1 | `src/scraper.py` | pandas.pydata.org → `data/raw/*.txt` |
| 2 | `src/qa_generator*.py` | raw text → `data/processed/qa_dataset.csv` |
| 3 | `src/vector_store.py` | CSV + raw text → `chroma_db/{docs,qa}/` |
| 4 | `src/retriever.py` + `src/chatbot.py` | query → grounded answer |
| 5 | `app.py` | Streamlit UI |

**Hybrid retrieval** (core logic in `src/retriever.py`):
- Stage 1: semantic search in QA vector store — if score ≥ `HYBRID_THRESHOLD` (0.75), return cached answer directly (no LLM call)
- Stage 2 fallback: top-5 doc chunks → Groq `llama-3.1-8b-instant` synthesizes answer

**Two ChromaDB collections** in `chroma_db/`:
- `qa/` — embedded Q&A pairs (fast path)
- `docs/` — embedded doc chunks (fallback path)

All constants live in `config.py`. Never hard-code paths, model names, or thresholds elsewhere.

## Conventions

- **Constants**: all in `config.py` (SCREAMING_SNAKE_CASE)
- **Functions/methods**: snake_case; **Classes**: PascalCase
- **Scraped files**: `docs__user_guide__{section}.txt`
- **Q&A CSVs**: 3 columns — `question`, `answer`, `source_page`
- Each `src/` module is independently runnable via `if __name__ == "__main__"`
- Generators are **resumable**: they skip already-processed `source_page` entries; do not restart a generator mid-run unless you delete the output CSV first
- Groq rate limiting: 429 errors trigger 60–150s backoff — do not reduce sleep intervals
- Vector store batches embeds in chunks of 500 — keep this when modifying `vector_store.py`

## Key Files

- [`config.py`](config.py) — all settings, thresholds, model names, paths
- [`src/retriever.py`](src/retriever.py) — hybrid retrieval logic
- [`src/chatbot.py`](src/chatbot.py) — RAG orchestrator
- [`src/vector_store.py`](src/vector_store.py) — ChromaDB collection builders
- [`README.md`](README.md) — full setup walkthrough

## Common Pitfalls

- **ChromaDB already populated**: `vector_store.py` will raise if collections exist; delete `chroma_db/` or use `get_or_create_collection` before re-running
- **Missing `.env`**: `config.py` loads it via `python-dotenv`; the app silently fails without `GROQ_API_KEY`
- **Ollama not running**: `qa_generator_ollama.py` and `test_ollama.py` require `ollama serve` on port 11434 first
- **Streamlit port conflict**: default port is 8501; use `streamlit run app.py --server.port 8502` if occupied
