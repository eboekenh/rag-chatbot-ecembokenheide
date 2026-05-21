import os
import sys
import functools

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RAW_DATA_DIR,
    QA_DATASET_PATH,
    CHROMA_PERSIST_DIR,
    DOC_COLLECTION_NAME,
    QA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


@functools.lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_raw_pages(raw_dir: str) -> list[dict]:
    pages = []
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(raw_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

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
            pages.append({"url": url, "title": title, "content": content})

    return pages


def chunk_documents(pages: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = []
    for page in pages:
        chunks = splitter.split_text(page["content"])
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"source": page["url"], "title": page["title"]},
            ))
    return docs


def build_doc_store(raw_dir: str = None, persist_dir: str = None) -> Chroma:
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR
    if persist_dir is None:
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, "docs")

    pages = load_raw_pages(raw_dir)
    if not pages:
        raise FileNotFoundError(f"No .txt files in {raw_dir}. Run scraper.py first.")

    print(f"Chunking {len(pages)} pages...")
    docs = chunk_documents(pages)
    print(f"Created {len(docs)} chunks")

    embedder = get_embedder()

    print(f"Building doc store → {persist_dir}")
    store = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name=DOC_COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    store.persist()
    print(f"Doc store ready: {len(docs)} chunks indexed")
    return store


def build_qa_store(qa_csv: str = None, persist_dir: str = None) -> Chroma:
    if qa_csv is None:
        qa_csv = QA_DATASET_PATH
    if persist_dir is None:
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, "qa")

    if not os.path.exists(qa_csv):
        raise FileNotFoundError(f"{qa_csv} not found. Run qa_generator.py first.")

    df = pd.read_csv(qa_csv)
    required = {"question", "answer", "source_page"}
    if not required.issubset(df.columns):
        raise ValueError(f"qa_dataset.csv must have columns: {required}")

    # Embed only the question – answer stored as metadata
    docs = []
    for _, row in df.iterrows():
        docs.append(Document(
            page_content=str(row["question"]),
            metadata={
                "answer": str(row["answer"]),
                "source": str(row["source_page"]),
            },
        ))

    embedder = get_embedder()

    print(f"Building Q&A store → {persist_dir} ({len(docs)} pairs)")
    store = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name=QA_COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    store.persist()
    print(f"Q&A store ready: {len(docs)} pairs indexed")
    return store


def load_doc_store(persist_dir: str = None) -> Chroma:
    if persist_dir is None:
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, "docs")
    return Chroma(
        collection_name=DOC_COLLECTION_NAME,
        embedding_function=get_embedder(),
        persist_directory=persist_dir,
    )


def load_qa_store(persist_dir: str = None) -> Chroma:
    if persist_dir is None:
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, "qa")
    return Chroma(
        collection_name=QA_COLLECTION_NAME,
        embedding_function=get_embedder(),
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    build_doc_store()
    build_qa_store()
    print("\nBoth vector stores built successfully.")
    print(f"Persisted to: {CHROMA_PERSIST_DIR}/")
