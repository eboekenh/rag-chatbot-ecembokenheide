"""RAG chatbot orchestrator for the pandas documentation assistant.

Exposes a single ``RAGChatbot`` class that ties together retrieval and
generation:

* **Stage 1 (qa_match)** – ``HybridRetriever`` finds a match in the
  pre-built QA store whose relevance score meets ``HYBRID_THRESHOLD``
  and returns the stored answer directly, without calling the LLM.
* **Stage 2 (doc_search)** – The retriever falls back to the document
  store, and the top-k chunks are forwarded to a Groq-hosted LLM to
  synthesise an answer.

Conversation history is kept in ``self.memory`` and injected into every
LLM prompt up to ``MAX_HISTORY_TURNS`` turns.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GROQ_API_KEY,
    GROQ_QA_MODEL,
    LLM_TEMPERATURE,
    MAX_HISTORY_TURNS,
    QA_DATASET_PATH,
    RAW_DATA_DIR,
)

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant specialized in the pandas library.
Answer based ONLY on the provided context. Be concise and precise.
If the context does not contain enough information, say: "I don't have enough information about that in the pandas documentation."
Always cite the source page at the end of your answer as: Source: <url>"""


class RAGChatbot:
    """Retrieval-augmented generation chatbot for pandas documentation queries.

    Combines a two-stage hybrid retriever with a Groq LLM and a rolling
    conversation memory window.

    Attributes
    ----------
    retriever : HybridRetriever
        Handles QA-store lookup and doc-store RAG fallback.
    llm : ChatGroq
        Groq-hosted LLM used for answer synthesis in doc_search mode.
    memory : list[dict]
        Conversation history as a list of ``{role, content}`` dicts,
        capped at ``MAX_HISTORY_TURNS * 2`` entries.
    """

    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_QA_MODEL,
            temperature=LLM_TEMPERATURE,
        )
        self.memory: list[dict] = []

    def _build_messages(self, query: str, context_docs: list[Document]) -> list[dict]:
        """Assemble the message list for a Groq LLM call.

        The list is ordered as: system prompt → recent memory turns →
        current user message (with retrieved context prepended).

        Parameters
        ----------
        query : str
            The user's current question.
        context_docs : list[Document]
            Retrieved document chunks to include as context.

        Returns
        -------
        list[dict]
            A list of ``{role, content}`` dicts ready for
            ``ChatGroq.invoke()``.
        """
        context_blocks = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_blocks.append(f"[{i}] (Source: {source})\n{doc.page_content}")
        context_text = "\n\n".join(context_blocks)

        user_content = f"""Context:
{context_text}

Question: {query}"""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add last N turns of memory
        recent_memory = self.memory[-(MAX_HISTORY_TURNS * 2):]
        messages.extend(recent_memory)

        messages.append({"role": "user", "content": user_content})
        return messages

    def chat(self, user_message: str) -> dict:
        """Process a user message and return an answer with metadata.

        Runs the two-stage retrieval pipeline and, in doc_search mode,
        calls the LLM to synthesise an answer from the retrieved chunks.
        Updates ``self.memory`` with the exchange regardless of outcome.

        Parameters
        ----------
        user_message : str
            The raw question from the user.

        Returns
        -------
        dict
            A dict with keys:

            * ``answer`` (str) – the generated or retrieved answer.
            * ``sources`` (list[str]) – source URLs used; empty on error
              or when no context was found.
            * ``mode`` (str) – ``"qa_match"`` or ``"doc_search"``.
            * ``confidence`` (float) – retrieval relevance score, rounded
              to three decimal places.
            * ``matched_question`` (str | None) – the matched QA-store
              question in qa_match mode, else ``None``.
        """
        retrieval = self.retriever.retrieve(user_message)
        logger.debug("Retrieval mode: %s (confidence=%.3f)", retrieval["mode"], retrieval["confidence"])

        if retrieval["mode"] == "qa_match":
            answer = retrieval["answer"]
            sources = [retrieval["source"]] if retrieval["source"] else []
            mode = "qa_match"
            confidence = retrieval["confidence"]
            matched_question = retrieval["matched_question"]
        else:
            context_docs = retrieval["context_docs"]

            if not context_docs:
                answer = "I don't have enough information about that in the pandas documentation."
                sources = []
            else:
                messages = self._build_messages(user_message, context_docs)
                try:
                    response = self.llm.invoke(messages)
                    answer = response.content
                    sources = list({
                        doc.metadata.get("source", "")
                        for doc in context_docs
                        if doc.metadata.get("source")
                    })
                except Exception as exc:
                    logger.error("LLM call failed: %s", exc)
                    answer = "I'm sorry, I couldn't reach the language model. Please try again."
                    sources = []

            mode = "doc_search"
            confidence = retrieval["confidence"]
            matched_question = None

        # Update memory
        self.memory.append({"role": "user", "content": user_message})
        self.memory.append({"role": "assistant", "content": answer})
        if len(self.memory) > MAX_HISTORY_TURNS * 2:
            self.memory = self.memory[-(MAX_HISTORY_TURNS * 2):]

        return {
            "answer": answer,
            "sources": sources,
            "mode": mode,
            "confidence": round(confidence, 3),
            "matched_question": matched_question,
        }

    def reset_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory = []

    def get_stats(self) -> dict:
        """Return best-effort runtime statistics about the chatbot's data.

        All three stat lookups are wrapped in silent try/except blocks;
        a failure in any one does not affect the others.

        Returns
        -------
        dict
            A dict with keys:

            * ``qa_pairs`` (int) – number of rows in the QA CSV dataset.
            * ``doc_chunks`` (int) – number of chunks in the doc vector
              store (via the internal ChromaDB collection API).
            * ``pages_scraped`` (int) – number of ``.txt`` files in the
              raw data directory.
            * ``memory_turns`` (int) – current number of conversation
              turns held in memory.
        """
        qa_count = 0
        doc_count = 0
        page_count = 0

        try:
            import pandas as pd
            if os.path.exists(QA_DATASET_PATH):
                qa_count = len(pd.read_csv(QA_DATASET_PATH))
        except Exception:
            pass

        try:
            if os.path.exists(RAW_DATA_DIR):
                page_count = len([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".txt")])
        except Exception:
            pass

        try:
            # _collection is an internal attribute of LangChain's Chroma wrapper.
            # No public API exists for a direct count; the try/except handles
            # any breakage from future LangChain/ChromaDB version changes.
            doc_count = self.retriever.doc_store._collection.count()
        except Exception:
            pass

        return {
            "qa_pairs": qa_count,
            "doc_chunks": doc_count,
            "pages_scraped": page_count,
            "memory_turns": len(self.memory) // 2,
        }
