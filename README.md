# 🐼 Pandas Docs RAG Chatbot

A RAG-powered chatbot that answers questions about the **pandas library** using the official [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html).

Built for **AGAI-03 Assignment 1** – Agentic AI Bootcamp Contest #1.

---

## What It Does

- Scrapes 23 pages from the pandas User Guide
- Generates 150+ synthetic Q&A pairs using an LLM
- Stores everything in a Chroma vector database
- Answers your pandas questions using **hybrid retrieval**:
  - First checks the Q&A dataset (fast, cached answers)
  - Falls back to full document search if no good match is found
- Clean Streamlit chat interface with source citations

---

## Project Structure

```
rag-qa-chatbot/
├── data/
│   ├── raw/                  # scraped pages (.txt)
│   ├── processed/
│   └── qa_dataset.csv        # generated Q&A pairs
├── src/
│   ├── scraper.py            # web scraper (Phase 1)
│   ├── qa_generator.py       # Q&A generation via LLM (Phase 2)
│   ├── vector_store.py       # Chroma DB setup (Phase 3)
│   ├── retriever.py          # hybrid retrieval logic (Phase 4)
│   └── chatbot.py            # RAG chatbot engine (Phase 4)
├── app.py                    # Streamlit UI (Phase 5)
├── config.py                 # all constants and settings
├── requirements.txt
├── report.pdf
└── .gitignore
```

---

## Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/eboekenh/rag-qa-chatbot.git
cd rag-qa-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

### 4. Scrape the pandas docs

```bash
python src/scraper.py
```

Scrapes 23 pages from pandas.pydata.org → saves to `data/raw/`

### 5. Generate Q&A pairs

```bash
python src/qa_generator.py
```

Generates ~150 Q&A pairs via Groq LLM → saves to `data/qa_dataset.csv`

### 6. Build the vector database

```bash
python src/vector_store.py
```

Creates two Chroma collections in `chroma_db/` (docs + Q&A)

### 7. Start the chatbot

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Scraping | requests, beautifulsoup4 |
| LLM | Groq – llama3-8b-8192 |
| Embeddings | sentence-transformers – all-MiniLM-L6-v2 |
| Vector DB | Chroma (2 collections) |
| Framework | LangChain, langchain-groq |
| UI | Streamlit |
| Data | pandas |

---

## How Hybrid Retrieval Works

```
User Question
      │
      ▼
Search Q&A Collection (cosine similarity)
      │
   Score ≥ 0.75? ──YES──► Return cached answer directly
      │
      NO
      │
      ▼
Search Document Collection (top-5 chunks)
      │
      ▼
LLM generates answer from context
      │
      ▼
Return answer + source citation
```

**Why two stages?**
- Q&A matching is fast and precise for questions already in the dataset
- Document search handles new/complex questions not covered by Q&A pairs

---

## Covered Topics

The chatbot can answer questions about:

- 10 Minutes to pandas
- Data structures (Series, DataFrame)
- Essential basic functionality
- Indexing and selecting data
- Merging, joining, concatenating
- GroupBy operations
- Time series & date functionality
- Handling missing data
- Text data operations
- Visualization
- I/O tools (CSV, Excel, JSON, SQL)
- And more...

---

## Submission

- **Course:** AGAI-03 – Agentic AI Bootcamp
- **Assignment:** Website-Specific RAG Chatbot
- **Deadline:** 29th May 2026
- **Contest:** Agentic AI Bootcamp Contest #1
