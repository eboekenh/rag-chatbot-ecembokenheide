import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Paths ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
QA_DATASET_PATH = "data/qa_dataset.csv"
CHROMA_PERSIST_DIR = "chroma_db"

# --- Chroma Collections ---
DOC_COLLECTION_NAME = "pandas_docs"
QA_COLLECTION_NAME = "pandas_qa"

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
HYBRID_THRESHOLD = 0.75
TOP_K_DOCS = 5

# --- Groq (used by qa_generator) ---
GROQ_QA_MODEL = "llama-3.1-8b-instant"

# --- LLM (Groq, used by chatbot) ---
LLM_TEMPERATURE = 0.3

# --- Chat Memory ---
MAX_HISTORY_TURNS = 6

# --- Scraper ---
SCRAPE_DELAY = 2  # seconds between requests

SCRAPE_TARGETS = [
    "https://pandas.pydata.org/docs/user_guide/10min.html",
    "https://pandas.pydata.org/docs/user_guide/dsintro.html",
    "https://pandas.pydata.org/docs/user_guide/basics.html",
    "https://pandas.pydata.org/docs/user_guide/io.html",
    "https://pandas.pydata.org/docs/user_guide/indexing.html",
    "https://pandas.pydata.org/docs/user_guide/advanced.html",
    "https://pandas.pydata.org/docs/user_guide/merging.html",
    "https://pandas.pydata.org/docs/user_guide/reshaping.html",
    "https://pandas.pydata.org/docs/user_guide/text.html",
    "https://pandas.pydata.org/docs/user_guide/missing_data.html",
    "https://pandas.pydata.org/docs/user_guide/duplicates.html",
    "https://pandas.pydata.org/docs/user_guide/categorical.html",
    "https://pandas.pydata.org/docs/user_guide/visualization.html",
    "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "https://pandas.pydata.org/docs/user_guide/window.html",
    "https://pandas.pydata.org/docs/user_guide/timeseries.html",
    "https://pandas.pydata.org/docs/user_guide/timedeltas.html",
    "https://pandas.pydata.org/docs/user_guide/options.html",
    "https://pandas.pydata.org/docs/user_guide/enhancingperf.html",
    "https://pandas.pydata.org/docs/user_guide/scale.html",
    "https://pandas.pydata.org/docs/user_guide/sparse.html",
    "https://pandas.pydata.org/docs/user_guide/gotchas.html",
    "https://pandas.pydata.org/docs/user_guide/cookbook.html",
]
