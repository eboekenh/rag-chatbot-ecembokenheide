import os
import re
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

DOC_SEARCH_MIN_CONFIDENCE = 0.20
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
_SYSTEM_PROMPT = """You are a helpful pandas expert assistant.

## Your job
Answer the user's CURRENT question using the Documentation Context provided below.
The Chat History is only for resolving pronouns like "it", "this", "that" — do NOT let previous topics override the current question.

## Rules
1. Always answer the CURRENT question. Ignore unrelated history.
2. Give a clear, detailed explanation — not just a one-liner.
3. Only include code examples if the Documentation Context explicitly contains them. Never invent output values.
4. Use simple language so beginners can understand.
5. If the Documentation Context does not contain enough information, say: "I don't have enough information about that in the pandas documentation."
6. End your answer with: Source: <url>"""


# ── Follow-up detection ───────────────────────────────────────────────────────
_PANDAS_KEYWORDS = {
    "dataframe", "series", "pandas", "column", "row", "index", "merge", "join",
    "group", "groupby", "sort", "filter", "read", "csv", "excel", "missing",
    "nan", "null", "dtype", "apply", "map", "concat", "pivot", "melt", "stack",
    "resample", "rolling", "shift", "diff", "cumsum", "describe", "info",
    "loc", "iloc", "assign", "drop", "rename", "fillna", "dropna", "replace",
    "astype", "datetime", "value_counts", "agg", "plot", "reset", "set_index",
    "mean", "median", "sum", "count", "max", "min", "std", "var", "unique",
    "duplicates", "sample", "head", "tail", "select", "subset", "condition",
    "boolean", "mask", "where", "query", "eval", "crosstab", "cut", "qcut",
}


def _is_followup(query: str) -> bool:
    """True if query is a short follow-up without pandas-specific terms."""
    words = query.lower().split()
    if len(words) >= 8:
        return False
    return not any(kw in query.lower() for kw in _PANDAS_KEYWORDS)


def _extract_source_url(answer: str) -> list[str]:
    """Extract the single Source URL the LLM cited in its answer."""
    match = re.search(r'[Ss]ource:\s*(https?://\S+)', answer)
    if match:
        return [match.group(1).rstrip('.,)>')]
    return []


_PANDAS_KEYWORDS = {
    "pandas", "dataframe", "df", "series", "column", "columns", "row", "rows",
    "index", "merge", "join", "concat", "groupby", "group", "sort", "filter",
    "read_csv", "read_excel", "to_csv", "iloc", "loc", "apply", "map", "lambda",
    "pivot", "melt", "stack", "unstack", "resample", "rolling", "shift", "diff",
    "fillna", "dropna", "isna", "isnull", "notnull", "duplicated", "drop_duplicates",
    "rename", "reset_index", "set_index", "astype", "dtypes", "dtype", "shape",
    "values", "head", "tail", "describe", "info", "value_counts", "unique",
    "nunique", "aggregate", "agg", "transform", "explode", "crosstab",
    "data", "dataset", "table", "cell", "missing", "nan", "null",
}


def _is_pandas_related(query: str) -> bool:
    words = set(query.lower().replace("?", "").replace(",", "").split())
    return bool(words & _PANDAS_KEYWORDS)


def _get_retrieval_query(query: str, memory: list) -> str:
    """For follow-up questions, prepend the last user topic to improve retrieval."""
    if not _is_followup(query):
        return query
    for msg in reversed(memory):
        if msg["role"] == "user":
            return f"{msg['content']} {query}"
    return query


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

    user_content = (
        f"## Documentation Context\n{context_text}\n\n"
        f"## Current Question\n{query}"
    )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(memory[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_content})
    return messages


def _build_qa_messages(query: str, qa_matches: list, memory: list) -> list:
    """QA-Match: Using several relevant Q&A pairs as context, Ollama explains in detail.

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
    is_followup = _is_followup(query)
    has_prior_context = is_followup and any(m["role"] == "user" for m in memory)
    if not _is_pandas_related(query) and not has_prior_context:
        return None, {"mode": "doc_search", "confidence": 0.0, "sources": [], "matched_question": None}

    retrieval_query = _get_retrieval_query(query, memory)
    retrieval = retriever.retrieve(retrieval_query)
    confidence = retrieval["confidence"]
    doc_confidence = retrieval.get("doc_confidence")

    if retrieval["mode"] == "qa_match":
        qa_matches = retrieval["qa_matches"]
        messages = _build_qa_messages(query, qa_matches, memory)
        seen = set()
        sources = []
        for doc, _ in qa_matches:
            src = doc.metadata.get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        mode = "qa_match"
        matched_question = retrieval["matched_question"] or ""
    else:
        context_docs = retrieval["context_docs"]
        matched_question = None
        has_prior_context = _is_followup(query) and any(m["role"] == "user" for m in memory)
        low_confidence = not context_docs or (doc_confidence or 0.0) < DOC_SEARCH_MIN_CONFIDENCE
        if low_confidence and not has_prior_context:
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

    display_confidence = round(confidence, 3) if mode == "qa_match" else round(doc_confidence or 0.0, 3)
    return messages, {
        "mode": mode,
        "confidence": display_confidence,
        "sources": sources,
        "matched_question": matched_question if mode == "qa_match" else None,
    }


def chat(query: str, retriever: HybridRetriever, llm, memory: list) -> dict:
    """Run hybrid retrieval + Ollama generation. Returns result dict."""
    retrieval_query = _get_retrieval_query(query, memory)
    retrieval = retriever.retrieve(retrieval_query)
    confidence = retrieval["confidence"]
    doc_confidence = retrieval.get("doc_confidence")

    if retrieval["mode"] == "qa_match":
        qa_matches = retrieval["qa_matches"]

        seen = set()
        sources = []
        for doc, _ in qa_matches:
            src = doc.metadata.get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
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
        has_prior_context = _is_followup(query) and any(m["role"] == "user" for m in memory)
        low_confidence = not context_docs or (doc_confidence or 0.0) < DOC_SEARCH_MIN_CONFIDENCE
        if low_confidence and not has_prior_context:
            answer = "I don't have enough information about that in the pandas documentation."
            sources = []
        else:
            messages = _build_messages(query, context_docs, memory)
            try:
                response = llm.invoke(messages)
                answer = response.content
                extracted = _extract_source_url(answer)
                sources = extracted if extracted else list({
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
        "ollama pull llama3.1:8b\n"
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
    st.caption("Local LLM · Ollama llama3.1:8b · RAG")
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
    qa_path = "data/processed/qa_dataset_ollama.csv"
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
2. Falls back to full doc search if no related QA is found

**Models:**
- LLM: Ollama llama3.1:8b
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
st.caption("Powered by RAG · Hybrid Q&A + Document Retrieval · Ollama llama3.1:8b")

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

    # Pre-save user message so follow-up context is available even if streaming
    # is interrupted by the next user input before st.rerun() completes.
    st.session_state.memory.append({"role": "user", "content": user_input})

    # Prepare retrieval using prior turns only (exclude the just-added current message)
    messages, meta = _prepare_chat(user_input, retriever, st.session_state.memory[:-1])

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
                # Use the URL the LLM actually cited instead of all retrieved sources
                extracted = _extract_source_url(answer)
                if extracted:
                    meta["sources"] = extracted
            except Exception as exc:
                answer = f"Ollama error: {exc}"
                st.error(answer)

        # Show meta badges right after the answer
        cols = st.columns([1, 1, 2])
        if meta["mode"] == "qa_match":
            cols[0].success(f"✅ Q&A Match ({meta['confidence']:.0%})")
        else:
            cols[0].info(f"🔍 Doc Search ({meta['confidence']:.0%})")
        if meta.get("matched_question"):
            with st.expander("🔗 Matched Q&A"):
                st.caption(f"**Matched question:** {meta['matched_question']}")

    # Save assistant response and trim memory
    st.session_state.memory.append({"role": "assistant", "content": answer})
    if len(st.session_state.memory) > MAX_HISTORY_TURNS * 2:
        del st.session_state.memory[:2]

    # Save to history for replay on next rerun
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    st.rerun()
