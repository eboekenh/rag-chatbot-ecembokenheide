"""Ollama-powered RAG chatbot Streamlit app.

Identical retrieval pipeline as app.py (hybrid QA-store + doc-store),
but uses a local Ollama LLM instead of the Groq API.

Run with:
    streamlit run app_ollama.py

Requirements:
    ollama serve          # must be running on localhost:11434
    ollama pull llama3.2:3b
"""
import os
import socket
import sys

import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Pandas Ollama Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_QA_MODEL,
    LLM_TEMPERATURE,
    MAX_HISTORY_TURNS,
    HYBRID_THRESHOLD,
)
from src.retriever import HybridRetriever
from langchain_core.documents import Document


# ── Ollama availability check ─────────────────────────────────────────────────
def _ollama_running() -> bool:
    """Return True if Ollama is reachable on localhost:11434."""
    try:
        host = OLLAMA_BASE_URL.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(OLLAMA_BASE_URL.rsplit(":", 1)[-1]) if ":" in OLLAMA_BASE_URL.rsplit("/", 1)[-1] else 11434
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a helpful pandas expert assistant. Use the provided context to answer questions.

Rules:
1. Always give a clear, detailed explanation — not just a one-liner.
2. Always include at least one practical Python/pandas code example using a code block.
3. If the concept has multiple use cases, show 2-3 short examples.
4. Use simple language so beginners can understand.
5. If the context does not contain enough information, say: "I don't have enough information about that in the pandas documentation."
6. End your answer with: Source: <url>"""


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retriever (embedding model + vector stores)...")
def load_retriever() -> HybridRetriever:
    return HybridRetriever()


@st.cache_resource(show_spinner="Loading Ollama LLM...")
def load_llm():
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_QA_MODEL,
        temperature=LLM_TEMPERATURE,
    )


# ── Core chat function ────────────────────────────────────────────────────────
def _build_messages(query: str, context_docs: list, memory: list) -> list:
    context_blocks = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)

    user_content = f"Context:\n{context_text}\n\nQuestion: {query}"

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(memory[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_content})
    return messages


def _build_qa_messages(query: str, qa_matches: list, memory: list) -> list:
    """QA-Match: mehrere passende Q&A-Paare als Kontext, Ollama erklärt ausführlich.

    qa_matches: list of (Document, score) tuples, alle >= HYBRID_THRESHOLD
    """
    context_blocks = []
    for i, (doc, score) in enumerate(qa_matches, 1):
        q = doc.page_content
        a = doc.metadata.get("answer", "")
        src = doc.metadata.get("source", "")
        context_blocks.append(
            f"[Match {i} | Score: {score:.2f} | Source: {src}]\n"
            f"Q: {q}\n"
            f"A: {a}"
        )
    context_text = "\n\n".join(context_blocks)
    user_content = f"Context (top relevant Q&A pairs from documentation):\n{context_text}\n\nQuestion: {query}"
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(memory[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_content})
    return messages


def _prepare_chat(query: str, retriever: HybridRetriever, memory: list) -> tuple:
    """Return (messages_or_None, meta_dict) without calling the LLM."""
    retrieval = retriever.retrieve(query)
    confidence = retrieval["confidence"]

    if retrieval["mode"] == "qa_match":
        all_qa = retriever.search_qa_store(query, top_k=3)
        qa_matches = [(doc, score) for doc, score in all_qa if score >= HYBRID_THRESHOLD]
        if not qa_matches:
            qa_matches = [all_qa[0]] if all_qa else []
        messages = _build_qa_messages(query, qa_matches, memory)
        sources = list({
            doc.metadata.get("source", "")
            for doc, _ in qa_matches
            if doc.metadata.get("source")
        })
        mode = "qa_match"
        matched_question = retrieval["matched_question"] or ""
    else:
        context_docs = retrieval["context_docs"]
        matched_question = None
        if not context_docs:
            messages = None
            sources = []
        else:
            messages = _build_messages(query, context_docs, memory)
            sources = list({
                doc.metadata.get("source", "")
                for doc in context_docs
                if doc.metadata.get("source")
            })
        mode = "doc_search"

    return messages, {
        "mode": mode,
        "confidence": round(confidence, 3),
        "sources": sources,
        "matched_question": matched_question if mode == "qa_match" else None,
    }


def chat(query: str, retriever: HybridRetriever, llm, memory: list) -> dict:
    """Run hybrid retrieval + Ollama generation. Returns result dict."""
    retrieval = retriever.retrieve(query)
    confidence = retrieval["confidence"]

    if retrieval["mode"] == "qa_match":
        # Retrieve top-3 QA matches and keep those above the threshold
        all_qa = retriever.search_qa_store(query, top_k=3)
        qa_matches = [(doc, score) for doc, score in all_qa if score >= HYBRID_THRESHOLD]
        if not qa_matches:  # safety fallback
            qa_matches = [all_qa[0]] if all_qa else []

        # Collect sources from all matched pairs
        sources = list({
            doc.metadata.get("source", "")
            for doc, _ in qa_matches
            if doc.metadata.get("source")
        })
        mode = "qa_match"
        matched_question = retrieval["matched_question"] or ""

        messages = _build_qa_messages(query, qa_matches, memory)
        try:
            response = llm.invoke(messages)
            answer = response.content
        except Exception as exc:
            answer = retrieval["answer"]  # fallback: gespeicherte Kurzantwort
    else:
        context_docs = retrieval["context_docs"]
        matched_question = None
        if not context_docs:
            answer = "I don't have enough information about that in the pandas documentation."
            sources = []
        else:
            messages = _build_messages(query, context_docs, memory)
            try:
                response = llm.invoke(messages)
                answer = response.content
                sources = list({
                    doc.metadata.get("source", "")
                    for doc in context_docs
                    if doc.metadata.get("source")
                })
            except Exception as exc:
                answer = f"Ollama error: {exc}"
                sources = []

        mode = "doc_search"

    # Update memory
    memory.append({"role": "user", "content": query})
    memory.append({"role": "assistant", "content": answer})
    if len(memory) > MAX_HISTORY_TURNS * 2:
        del memory[:2]

    return {
        "answer": answer,
        "sources": sources,
        "mode": mode,
        "confidence": round(confidence, 3),
        "matched_question": matched_question,
    }


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []


# ── Ollama guard — show error before loading heavy resources ──────────────────
if not _ollama_running():
    st.error(
        "**Ollama is not running.**\n\n"
        "Please start it first:\n"
        "```powershell\n"
        "ollama serve\n"
        "ollama pull llama3.2:3b\n"
        "```\n"
        "Then reload this page.",
        icon="🦙",
    )
    st.stop()


# ── Load resources (cached after first run) ───────────────────────────────────
retriever = load_retriever()
llm = load_llm()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦙 Pandas Ollama Chatbot")
    st.caption("Local LLM · Ollama llama3.2:3b · RAG")
    st.divider()

    st.markdown("**Website:**")
    st.markdown("[pandas User Guide ↗](https://pandas.pydata.org/docs/user_guide/index.html)")
    st.divider()

    # Ollama status
    st.markdown("**🟢 Ollama Status**")
    st.success(f"Connected · {OLLAMA_QA_MODEL}")
    st.divider()

    # Sample questions
    st.markdown("**💡 Sample Questions**")
    sample_questions = [
        "How do I create a DataFrame from a dictionary?",
        "What is the difference between merge and join?",
        "How do I handle missing values in pandas?",
        "How does groupby work in pandas?",
        "How do I filter rows based on a condition?",
        "What is a MultiIndex?",
        "How do I sort a DataFrame?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"s_{q[:20]}"):
            st.session_state.pending_input = q

    st.divider()

    # Dataset sample viewer
    qa_path = "data/processed/qa_dataset_ollama - Kopie (2).csv"
    if not os.path.exists(qa_path):
        qa_path = "data/processed/qa_dataset.csv"
    if os.path.exists(qa_path):
        with st.expander("📄 View Sample Q&A Pairs"):
            try:
                df = pd.read_csv(qa_path)
                for _, row in df.sample(min(5, len(df))).reset_index(drop=True).iterrows():
                    st.markdown(f"**Q:** {row['question']}")
                    st.caption(f"A: {str(row['answer'])[:200]}{'...' if len(str(row['answer'])) > 200 else ''}")
                    st.divider()
            except Exception as e:
                st.error(f"Could not load Q&A file: {e}")

    st.divider()

    with st.expander("ℹ️ About"):
        st.markdown("""
**Hybrid Retrieval:**
1. Searches Q&A dataset first (cosine similarity)
2. Falls back to full doc search if score < 0.75

**Models:**
- LLM: Ollama llama3.2:3b (local)
- Embeddings: all-MiniLM-L6-v2

**Vector DB:** Chroma (2 collections)
        """)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🦙 Ask me anything about pandas")
st.caption("Powered by RAG · Hybrid Q&A + Document Retrieval · Ollama llama3.2:3b (local)")

if not st.session_state.messages:
    st.info(
        "👋 Hi! I can answer questions about the **pandas library** based on the official User Guide. "
        "I run **entirely on your local machine** via Ollama — no API key needed.\n\n"
        "Try asking: *How do I read a CSV file?* or *What is a DataFrame?*"
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            cols = st.columns([1, 1, 2])
            if meta.get("mode") == "qa_match":
                cols[0].success(f"✅ Q&A Match ({meta['confidence']:.0%})")
            else:
                cols[0].info(f"🔍 Doc Search ({meta['confidence']:.0%})")
            if meta.get("sources"):
                with cols[2].expander("📄 Sources"):
                    for src in meta["sources"]:
                        st.caption(f"• {src}")
            if meta.get("matched_question"):
                with st.expander("🔗 Matched Q&A"):
                    st.caption(f"**Matched question:** {meta['matched_question']}")

# Handle sample question button clicks
pending = st.session_state.pop("pending_input", None)

# Chat input
user_input = st.chat_input("Ask about pandas...") or pending

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare retrieval (fast) — no LLM call yet
    messages, meta = _prepare_chat(user_input, retriever, st.session_state.memory)

    # Stream the assistant response token-by-token
    with st.chat_message("assistant"):
        if messages is None:
            answer = "I don't have enough information about that in the pandas documentation."
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(
                    chunk.content for chunk in llm.stream(messages)
                )
            except Exception as exc:
                answer = f"Ollama error: {exc}"
                st.error(answer)

        # Show meta badges right after the answer
        cols = st.columns([1, 1, 2])
        if meta["mode"] == "qa_match":
            cols[0].success(f"✅ Q&A Match ({meta['confidence']:.0%})")
        else:
            cols[0].info(f"🔍 Doc Search ({meta['confidence']:.0%})")
        if meta.get("sources"):
            with cols[2].expander("📄 Sources"):
                for src in meta["sources"]:
                    st.caption(f"• {src}")
        if meta.get("matched_question"):
            with st.expander("🔗 Matched Q&A"):
                st.caption(f"**Matched question:** {meta['matched_question']}")

    # Update conversation memory
    st.session_state.memory.append({"role": "user", "content": user_input})
    st.session_state.memory.append({"role": "assistant", "content": answer})
    if len(st.session_state.memory) > MAX_HISTORY_TURNS * 2:
        del st.session_state.memory[:2]

    # Save to history for replay on next rerun
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    st.rerun()
