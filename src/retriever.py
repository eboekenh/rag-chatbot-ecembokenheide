import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import HYBRID_THRESHOLD, TOP_K_DOCS

from langchain.schema import Document
from src.vector_store import load_doc_store, load_qa_store


class HybridRetriever:
    def __init__(self):
        print("Loading vector stores...")
        self.doc_store = load_doc_store()
        self.qa_store = load_qa_store()
        print("Retriever ready.")

    def search_qa_store(self, query: str, top_k: int = 1) -> list[tuple[Document, float]]:
        results = self.qa_store.similarity_search_with_relevance_scores(query, k=top_k)
        return results

    def search_doc_store(self, query: str, top_k: int = None) -> list[Document]:
        if top_k is None:
            top_k = TOP_K_DOCS
        return self.doc_store.similarity_search(query, k=top_k)

    def retrieve(self, query: str) -> dict:
        qa_results = self.search_qa_store(query, top_k=1)
        best_score = qa_results[0][1] if qa_results else 0.0
        best_doc = qa_results[0][0] if qa_results else None

        if best_score >= HYBRID_THRESHOLD and best_doc is not None:
            return {
                "mode": "qa_match",
                "answer": best_doc.metadata["answer"],
                "source": best_doc.metadata["source"],
                "matched_question": best_doc.page_content,
                "confidence": best_score,
                "context_docs": [],
            }

        context_docs = self.search_doc_store(query)
        return {
            "mode": "doc_search",
            "answer": None,
            "source": None,
            "matched_question": None,
            "confidence": best_score,
            "context_docs": context_docs,
        }
