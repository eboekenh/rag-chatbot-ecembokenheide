# Pandas Docs RAG Chatbot

A website-specific Retrieval-Augmented Generation (RAG) chatbot for answering questions about the official [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html).

This project was built for **AGAI-03 Assignment 1: Website-Specific RAG Chatbot using Scraping + Hybrid Retrieval**. It demonstrates the full pipeline required by the assignment: scraping public web content, generating synthetic Q&A pairs, building vector databases, implementing hybrid retrieval, and exposing the system through a Streamlit chat interface.

## What The Project Does

- Scrapes 23 important pages from the pandas User Guide.
- Cleans documentation text by removing navigation, sidebars, footers, scripts, and other non-content elements.
- Generates synthetic Q&A pairs from the scraped documentation using Groq or a local Ollama model.
- Builds two persistent ChromaDB vector stores:
  - `docs`: chunked pandas documentation pages.
  - `qa`: generated Q&A pairs, with questions embedded and answers stored as metadata.
- Answers user questions with hybrid retrieval:
  - First searches the Q&A vector store for a confident semantic match.
  - Falls back to document chunk retrieval and LLM answer synthesis when no Q&A match is strong enough.
- Provides a Streamlit chat UI with source citations, matched Q&A details, sidebar project stats, sample questions, and a clear-chat button.
- Includes an Ollama-powered local variant and Selenium screenshot validation workflow.

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.10+ |
| Web scraping | `requests`, `beautifulsoup4`, `lxml` |
| Data processing | `pandas`, `tqdm` |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| RAG framework | LangChain |
| Hosted LLM | Groq `llama-3.1-8b-instant` |
| Local LLM variant | Ollama `llama3.2:3b` |
| UI | Streamlit |
| UI validation | Selenium, Chrome WebDriver |

## Repository Structure

```text
rag-qa-chatbot/
├── app.py                         # Main Groq-powered Streamlit app
├── app_ollama.py                  # Local Ollama-powered Streamlit app
├── config.py                      # Central settings, paths, thresholds, models
├── requirements.txt               # Python dependencies
├── project_report.docx            # Assignment report
├── data/
│   ├── raw/                       # 23 scraped pandas documentation pages
│   ├── processed/                 # Processed Q&A datasets
│   ├── qa_dataset.csv             # Larger generated Q&A dataset copy
│   └── qa_dataset_v1.csv          # Dataset version copy
├── chroma_db/
│   ├── docs/                      # Persisted document chunk vector store
│   └── qa/                        # Persisted Q&A vector store
├── screenshots_ollama/            # Selenium screenshots from UI validation
├── src/
│   ├── scraper.py                 # Phase 1: scrape and clean pandas docs
│   ├── qa_generator.py            # Phase 2: Groq Q&A generation
│   ├── qa_generator_ollama.py     # Phase 2 variant: Ollama Q&A generation
│   ├── qa_generator_csv.py        # Phase 2 variant: generate from CSV input
│   ├── vector_store.py            # Phase 3: build/load Chroma vector stores
│   ├── retriever.py               # Phase 4: two-stage hybrid retrieval
│   └── chatbot.py                 # Phase 4: RAG orchestration with Groq
├── test_ollama.py                 # Ollama connectivity check
├── test_ollama_chatbot.py         # Streamlit comparison of Q&A datasets
└── test_app_ollama_selenium.py    # Selenium UI screenshot workflow
```

## Current Project Data

| Asset | Current Count / Value |
|---|---:|
| Scraped raw documentation pages | 23 |
| Selenium screenshots in `screenshots_ollama/` | 49 |
| `data/processed/qa_dataset.csv` rows | 19 |
| `data/processed/qa_dataset_ollama.csv` rows | 1347 |
| `data/processed/qa_dataset_ollama - Kopie (2).csv` rows | 424 |
| `data/qa_dataset.csv` rows | 2789 |
| `data/qa_dataset_v1.csv` rows | 2789 |

The active default Q&A path in `config.py` is `data/processed/qa_dataset_ollama.csv`.

## Configuration

All important settings live in `config.py`.

| Setting | Value |
|---|---|
| `RAW_DATA_DIR` | `data/raw` |
| `PROCESSED_DATA_DIR` | `data/processed` |
| `QA_DATASET_PATH` | `data/processed/qa_dataset_ollama.csv` |
| `CHROMA_PERSIST_DIR` | `chroma_db` |
| `DOC_COLLECTION_NAME` | `pandas_docs` |
| `QA_COLLECTION_NAME` | `pandas_qa` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | `1000` |
| `CHUNK_OVERLAP` | `200` |
| `HYBRID_THRESHOLD` | `0.75` |
| `TOP_K_DOCS` | `5` |
| `GROQ_QA_MODEL` | `llama-3.1-8b-instant` |
| `OLLAMA_QA_MODEL` | `llama3.2:3b` |
| `MAX_HISTORY_TURNS` | `6` |

Do not hard-code these values in other files. Update `config.py` instead.

## Setup

### 1. Create And Activate The Environment

A `.venv` is already present in the project root. On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or from bash/cmd:

```bash
source .venv/Scripts/activate
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

### 2. Configure Groq

Create a `.env` file in the project root:

```env
GROQ_API_KEY=<your_groq_api_key>
```

Do not commit `.env`. The project loads it through `python-dotenv`.

### 3. Optional: Configure Ollama

The local app and Ollama Q&A generator require Ollama running on port `11434`:

```powershell
ollama serve
ollama pull llama3.2:3b
```

## Build The Full Pipeline

Run these commands from the project root.

### Phase 1: Scrape The pandas User Guide

```powershell
python src/scraper.py
```

This downloads and cleans 23 pandas documentation pages into `data/raw/`.

### Phase 2: Generate Synthetic Q&A Pairs

Groq generator:

```powershell
python src/qa_generator.py
```

Ollama generator:

```powershell
python src/qa_generator_ollama.py
```

CSV-based generator:

```powershell
python src/qa_generator_csv.py
```

The generators are resumable. They skip already processed `source_page` entries in the output CSV. If you want a completely fresh dataset, delete the target CSV first.

### Phase 3: Build The Vector Stores

```powershell
python src/vector_store.py
```

This builds:

- `chroma_db/docs/` from chunked raw documentation.
- `chroma_db/qa/` from generated Q&A pairs.

Embedding is batched in groups of 500 documents to avoid unnecessary memory pressure.

### Phase 4 And 5: Run The Chatbot

Groq-powered app:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Ollama-powered local app:

```powershell
streamlit run app_ollama.py --server.port 8503
```

Open:

```text
http://localhost:8503
```

## How Hybrid Retrieval Works

The chatbot uses a two-stage retrieval strategy implemented in `src/retriever.py`.

```text
User question
      |
      v
Search Q&A vector store
      |
      | score >= HYBRID_THRESHOLD (0.75)
      |---- yes --> return matched stored answer / Q&A context
      |
      no
      |
      v
Search document vector store
      |
      v
Retrieve top 5 documentation chunks
      |
      v
LLM synthesizes a grounded answer
      |
      v
Return answer + source citations + retrieval metadata
```

### Stage 1: Q&A Semantic Match

The Q&A collection embeds generated questions. Answers and source URLs are stored in metadata. If a user question is semantically close enough to an existing generated question, the system can answer quickly from the Q&A store.

In the Groq chatbot path, a confident Q&A hit returns the stored answer directly. In the Ollama app, the top matching Q&A pairs can be passed to the local model so it can produce a fuller explanation while still staying grounded in generated Q&A content.

### Stage 2: Document Search Fallback

When the Q&A score is below `0.75`, the system searches the full documentation vector store and sends the top 5 chunks to the LLM. The system prompt instructs the model to answer only from the provided context and cite the source page.

## Streamlit UI Features

The main app includes:

- Chat-style input and message history.
- Sidebar with project information and dataset statistics.
- Link to the pandas User Guide.
- Sample question buttons.
- Q&A sample viewer when a dataset file is available.
- Retrieval mode badges: Q&A match or document search.
- Confidence display.
- Source citation expander.
- Matched Q&A display for Q&A hits.
- Clear Chat button.

The Ollama app adds:

- Local Ollama availability guard.
- Token streaming through `st.write_stream`.
- Local model status in the sidebar.

## Testing And Validation

### Ollama Connectivity

```powershell
python test_ollama.py
```

This checks whether Ollama is reachable on `localhost:11434`.

### Q&A Dataset Comparison

```powershell
streamlit run test_ollama_chatbot.py
```

This compares the original Q&A store with an Ollama-generated Q&A dataset across 10 representative pandas questions.

### Streamlit UI Screenshot Test

Start the Ollama app first:

```powershell
streamlit run app_ollama.py --server.port 8503
```

Then run:

```powershell
python test_app_ollama_selenium.py
```

The Selenium test sends 10 main questions and 10 follow-up questions to the app, then saves screenshots to `screenshots_ollama/`.

## Covered pandas Topics

The scraped documentation covers key pandas User Guide sections, including:

- 10 minutes to pandas
- Series and DataFrame introduction
- Essential basic functionality
- Indexing and selecting data
- MultiIndex and advanced indexing
- Merge, join, and concatenate
- Reshaping and pivot tables
- Text data
- Missing data
- Duplicate labels
- Categorical data
- Visualization
- GroupBy
- Window operations
- Time series
- Timedeltas
- Options and settings
- Performance enhancement
- Scaling to large datasets
- Sparse data
- Gotchas
- Cookbook examples

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `GROQ_API_KEY not set` | Missing `.env` file | Create `.env` with `GROQ_API_KEY=...` |
| Streamlit app starts but cannot answer | Vector stores are missing | Run `python src/vector_store.py` |
| `No .txt files in data/raw` | Scraping has not been run | Run `python src/scraper.py` |
| Chroma collection already exists or stale data appears | Existing `chroma_db/` contents | Delete `chroma_db/` and rebuild if you need a fresh store |
| Ollama app shows not running | Ollama server is stopped | Run `ollama serve` and reload |
| Ollama model missing | `llama3.2:3b` not pulled | Run `ollama pull llama3.2:3b` |
| Streamlit port conflict | Default port already in use | Use `streamlit run app.py --server.port 8502` |
| Selenium test cannot find chat input | App not running or wrong port | Start `app_ollama.py` on port `8503` |

## Limitations

- Retrieval quality depends on the generated Q&A dataset and the threshold value.
- The `0.75` threshold is practical but not formally optimized.
- The scraped source is limited to selected pandas User Guide pages, not the entire pandas website.
- Local Ollama responses may be slower and less detailed than hosted models depending on hardware.
- The chatbot is designed for pandas documentation questions, not general Python or unrelated data science questions.
- ChromaDB stores are local and need rebuilding after major dataset changes.

## Future Improvements

- Add systematic retrieval evaluation with labeled queries and expected sources.
- Tune `HYBRID_THRESHOLD`, chunk size, and top-k values using evaluation results.
- Add automated tests for scraper output shape, Q&A CSV validation, and retriever behavior.
- Add a one-command setup script for rebuilding the full pipeline.
- Improve source presentation with chunk titles and section names.
- Add support for refreshing only changed documentation pages.
- Export chat logs for assignment demos and qualitative analysis.

## Assignment Notes

- **Course:** AGAI-03 Agentic AI Bootcamp
- **Assignment:** Website-Specific RAG Chatbot using Scraping + Hybrid Retrieval
- **Website:** pandas User Guide
- **Submission deadline:** 29 May 2026
- **Primary deliverables:** code repository, detailed documentation, Q&A dataset, scraped data, Streamlit app, project report
