"""
test_ollama_chatbot.py
-----------------------
Streamlit UI: compares retrieval quality between the original QA dataset
and the Ollama-generated QA dataset side by side.

Usage:
    streamlit run test_ollama_chatbot.py
"""

import os
import sys
import shutil
import logging

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from config import CHROMA_PERSIST_DIR, HYBRID_THRESHOLD, EMBEDDING_MODEL
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.vector_store import get_embedder

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QA Dataset Comparison",
    page_icon="🔍",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
ORIGINAL_QA_CSV = "data/processed/qa_dataset.csv"
OLLAMA_QA_CSV   = "data/processed/qa_dataset_ollama - Kopie (2).csv"
TEMP_STORE_DIR  = os.path.join(CHROMA_PERSIST_DIR, "_test_ollama_qa")

TEST_QUESTIONS = [
    "How do I read a CSV file into a DataFrame?",
    "What is the difference between loc and iloc?",
    "How can I group data and calculate aggregates?",
    "How do I handle missing values in pandas?",
    "How do I merge two DataFrames?",
    "What is a MultiIndex and when should I use it?",
    "How do I filter rows based on a condition?",
    "How can I apply a function to each row?",
    "How do I sort a DataFrame by column values?",
    "How do I plot a DataFrame?",
]


# ── Cached resource loaders ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder_cached():
    return get_embedder()


@st.cache_resource(show_spinner="Loading original QA store...")
def load_original_store():
    qa_dir = os.path.join(CHROMA_PERSIST_DIR, "qa")
    embedder = get_embedder_cached()
    return Chroma(
        collection_name="pandas_qa",
        embedding_function=embedder,
        persist_directory=qa_dir,
    )


@st.cache_resource(show_spinner="Building Ollama QA store from CSV (may take ~30s)...")
def build_ollama_store():
    if not os.path.exists(OLLAMA_QA_CSV):
        st.error(f"CSV not found: {OLLAMA_QA_CSV}")
        st.stop()

    df = pd.read_csv(OLLAMA_QA_CSV).dropna(subset=["question", "answer"])
    embedder = get_embedder_cached()

    docs = [
        Document(
            page_content=str(row["question"]),
            metadata={
                "answer": str(row["answer"]),
                "source": str(row.get("source_page", "")),
            },
        )
        for _, row in df.iterrows()
    ]

    # Remove stale temp store before building (ignore if locked by previous run)
    if os.path.exists(TEMP_STORE_DIR):
        try:
            shutil.rmtree(TEMP_STORE_DIR)
        except PermissionError:
            pass

    store = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name="test_ollama_qa",
        persist_directory=TEMP_STORE_DIR,
    )
    return store, len(df)


def search_store(store: Chroma, query: str) -> tuple[str, float, str]:
    results = store.similarity_search_with_relevance_scores(query, k=1)
    if not results:
        return "No match found", 0.0, ""
    doc, score = results[0]
    return doc.metadata.get("answer", ""), round(score, 3), doc.page_content


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("QA Dataset Comparison")
st.caption("Original Groq-generated dataset vs. Ollama-generated dataset — side by side")

original_store = load_original_store()
ollama_store, ollama_count = build_ollama_store()

c1, c2, c3 = st.columns(3)
c1.metric("Original store vectors", original_store._collection.count())
c2.metric("Ollama store vectors", ollama_count)
c3.metric("Hit threshold", HYBRID_THRESHOLD)

st.divider()

# ── Run comparisons ────────────────────────────────────────────────────────────
st.subheader(f"Testing {len(TEST_QUESTIONS)} questions")

rows = []
progress = st.progress(0, text="Running comparisons...")
for i, question in enumerate(TEST_QUESTIONS):
    o_answer, o_score, o_matched = search_store(original_store, question)
    n_answer, n_score, n_matched = search_store(ollama_store, question)
    rows.append({
        "Question":         question,
        "Orig score":       o_score,
        "Orig hit":         o_score >= HYBRID_THRESHOLD,
        "Orig matched Q":   o_matched[:80],
        "Orig answer":      o_answer[:200],
        "Ollama score":     n_score,
        "Ollama hit":       n_score >= HYBRID_THRESHOLD,
        "Ollama matched Q": n_matched[:80],
        "Ollama answer":    n_answer[:200],
        "Winner": "Ollama" if n_score > o_score else ("Original" if o_score > n_score else "Tie"),
    })
    progress.progress(
        (i + 1) / len(TEST_QUESTIONS),
        text=f"Q{i + 1}/{len(TEST_QUESTIONS)}: {question[:55]}...",
    )

progress.empty()
df = pd.DataFrame(rows)

# ── Results table ──────────────────────────────────────────────────────────────
st.dataframe(
    df[["Question", "Orig score", "Orig hit", "Ollama score", "Ollama hit", "Winner"]],
    use_container_width=True,
    column_config={
        "Orig score":   st.column_config.NumberColumn(format="%.3f"),
        "Ollama score": st.column_config.NumberColumn(format="%.3f"),
        "Orig hit":     st.column_config.CheckboxColumn("Orig hit"),
        "Ollama hit":   st.column_config.CheckboxColumn("Ollama hit"),
    },
)

# ── Score comparison chart ─────────────────────────────────────────────────────
st.subheader("Score comparison")
chart_df = df[["Question", "Orig score", "Ollama score"]].set_index("Question")
st.bar_chart(chart_df)

# ── Summary ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary")

n             = len(TEST_QUESTIONS)
orig_hits     = int(df["Orig hit"].sum())
ollama_hits   = int(df["Ollama hit"].sum())
ollama_better = int((df["Winner"] == "Ollama").sum())
orig_better   = int((df["Winner"] == "Original").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Original hits",          f"{orig_hits}/{n}",   f"{orig_hits   / n * 100:.0f}%")
m2.metric("Ollama hits",            f"{ollama_hits}/{n}", f"{ollama_hits  / n * 100:.0f}%")
m3.metric("Ollama scored higher",   f"{ollama_better}/{n}")
m4.metric("Original scored higher", f"{orig_better}/{n}")

if ollama_hits > orig_hits:
    st.success("Ollama dataset performs BETTER overall")
elif orig_hits > ollama_hits:
    st.warning("Original dataset performs better overall")
else:
    st.info("Datasets perform equally")

# ── Per-question detail expanders ──────────────────────────────────────────────
st.divider()
st.subheader("Per-question details")

for _, row in df.iterrows():
    winner_tag = f" — **{row['Winner']} wins**" if row["Winner"] != "Tie" else " — **Tie**"
    with st.expander(f"{row['Question']}{winner_tag}"):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Original")
            st.write(f"Score: `{row['Orig score']:.3f}` — {'**HIT**' if row['Orig hit'] else 'miss'}")
            st.caption(f"Matched: _{row['Orig matched Q']}..._")
            st.write(row["Orig answer"])
        with right:
            st.markdown("#### Ollama")
            st.write(f"Score: `{row['Ollama score']:.3f}` — {'**HIT**' if row['Ollama hit'] else 'miss'}")
            st.caption(f"Matched: _{row['Ollama matched Q']}..._")
            st.write(row["Ollama answer"])
