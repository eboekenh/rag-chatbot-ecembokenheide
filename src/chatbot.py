import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_HISTORY_TURNS,
    QA_DATASET_PATH,
    RAW_DATA_DIR,
)

from langchain.schema import Document
from langchain_groq import ChatGroq
from src.retriever import HybridRetriever

SYSTEM_PROMPT = """You are a helpful assistant specialized in the pandas library.
Answer based ONLY on the provided context. Be concise and precise.
If the context does not contain enough information, say: "I don't have enough information about that in the pandas documentation."
Always cite the source page at the end of your answer as: Source: <url>"""


class RAGChatbot:
    def __init__(self):
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set. Create a .env file with your key.")

        self.retriever = HybridRetriever()
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )
        self.memory: list[dict] = []

    def _build_messages(self, query: str, context_docs: list[Document]) -> list[dict]:
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
        retrieval = self.retriever.retrieve(user_message)

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
                response = self.llm.invoke(messages)
                answer = response.content
                sources = list({
                    doc.metadata.get("source", "")
                    for doc in context_docs
                    if doc.metadata.get("source")
                })

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
        self.memory = []

    def get_stats(self) -> dict:
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
            doc_count = self.retriever.doc_store._collection.count()
        except Exception:
            pass

        return {
            "qa_pairs": qa_count,
            "doc_chunks": doc_count,
            "pages_scraped": page_count,
            "memory_turns": len(self.memory) // 2,
        }
