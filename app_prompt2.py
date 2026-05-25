import os
import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Pandas Docs Chatbot",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached chatbot loader ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading chatbot... (first run may take a minute)")
def load_chatbot():
    from src.chatbot import RAGChatbot
    return RAGChatbot()


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chatbot" not in st.session_state:
    st.session_state.chatbot = load_chatbot()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🐼 Pandas RAG Chatbot")
    st.caption("AGAI-03 Assignment 1 · Agentic AI Bootcamp")
    st.divider()

    # Project info
    st.markdown("**Website:**")
    st.markdown("[pandas User Guide ↗](https://pandas.pydata.org/docs/user_guide/index.html)")
    st.divider()

    # Stats
    stats = st.session_state.chatbot.get_stats()
    st.markdown("**📊 Dataset Stats**")
    col1, col2 = st.columns(2)
    col1.metric("Q&A Pairs", stats["qa_pairs"])
    col2.metric("Doc Chunks", stats["doc_chunks"])
    col1.metric("Pages Scraped", stats["pages_scraped"])
    col2.metric("Memory Turns", stats["memory_turns"])
    st.divider()

    # Sample Q&As
    st.markdown("**💡 Sample Questions**")
    sample_questions = [
        "How do I create a DataFrame from a dictionary?",
        "What is the difference between merge and join in pandas?",
        "How do I handle missing values in pandas?",
        "How does groupby work in pandas?",
        "How do I filter rows based on a condition?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state.pending_input = q

    st.divider()

    # View sample Q&A pairs from dataset
    qa_path = "data/qa_dataset.csv"
    if os.path.exists(qa_path):
        with st.expander("📄 View Sample Q&A Pairs"):
            try:
                df = pd.read_csv(qa_path)
                sample = df.sample(min(5, len(df))).reset_index(drop=True)
                for _, row in sample.iterrows():
                    st.markdown(f"**Q:** {row['question']}")
                    st.caption(f"A: {row['answer'][:200]}{'...' if len(row['answer']) > 200 else ''}")
                    st.divider()
            except Exception as e:
                st.error(f"Could not load Q&A file: {e}")
    st.divider()

    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
**Hybrid Retrieval:**
1. Searches Q&A dataset first (cosine similarity)
2. Falls back to full doc search if score < 0.75

**Models:**
- LLM: Groq llama-3.1-8b-instant
- Embeddings: all-MiniLM-L6-v2

**Vector DB:** Chroma (2 collections)
        """)

    # Clear chat
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.chatbot.reset_memory()
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🐼 Ask me anything about pandas")
st.caption("Powered by RAG · Hybrid Q&A + Document Retrieval · Groq llama-3.1-8b-instant")

# Welcome message
if not st.session_state.messages:
    st.info(
        "👋 Hi! I can answer questions about the **pandas library** based on the official User Guide. "
        "Try asking: *How do I read a CSV file?* or *What is a DataFrame?*"
    )

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            cols = st.columns([1, 1, 2])

            # Mode badge
            if meta.get("mode") == "qa_match":
                cols[0].success(f"✅ Q&A Match ({meta['confidence']:.0%})")
            else:
                cols[0].info(f"🔍 Doc Search ({meta['confidence']:.0%})")

            # Sources
            if meta.get("sources"):
                with cols[2].expander("📄 Sources"):
                    for src in meta["sources"]:
                        st.caption(f"• {src}")

            # Matched question (only for qa_match)
            if meta.get("matched_question"):
                with st.expander("🔗 Matched Q&A"):
                    st.caption(f"**Matched question:** {meta['matched_question']}")


# ── Handle sample question button clicks ─────────────────────────────────────
pending = st.session_state.pop("pending_input", None)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about pandas...") or pending

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chatbot.chat(user_input)

        answer = result["answer"]
        st.markdown(answer)

        # Mode + confidence + sources
        cols = st.columns([1, 1, 2])
        if result["mode"] == "qa_match":
            cols[0].success(f"✅ Q&A Match ({result['confidence']:.0%})")
        else:
            cols[0].info(f"🔍 Doc Search ({result['confidence']:.0%})")

        if result.get("sources"):
            with cols[2].expander("📄 Sources"):
                for src in result["sources"]:
                    st.caption(f"• {src}")

        if result.get("matched_question"):
            with st.expander("🔗 Matched Q&A"):
                st.caption(f"**Matched question:** {result['matched_question']}")

    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "mode": result["mode"],
            "confidence": result["confidence"],
            "sources": result.get("sources", []),
            "matched_question": result.get("matched_question"),
        },
    })

    # Refresh stats in sidebar
    st.rerun()
