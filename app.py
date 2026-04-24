import streamlit as st
from core.extractor import extract_text
from core.chunker import chunk_text
from core.summarizer import summarize_chunks, final_summary
from utils.cleaner import clean_text

from core.embeddings import get_embeddings, model
from core.vector_store import create_faiss_index
from core.retriever import retrieve_chunks

st.title("📄 AI Document Summarizer")

# 🔥 Cache FAISS
@st.cache_data
def build_index(chunks):
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)
    return index

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing..."):
        text = extract_text("temp.pdf")
        text = clean_text(text)

        chunks = chunk_text(text)

        # 🔥 Safety fixes
        chunks = [c for c in chunks if len(c.split()) > 30]

        if len(chunks) == 0:
            st.error("No readable text found in PDF")
            st.stop()

        # 🔹 Summarization
        summary = summarize_chunks(chunks)
        final = final_summary(summary)

        # 🔹 FAISS (cached)
        index = build_index(chunks)

    st.subheader("✨ Final Summary")
    st.write(final)

    # 🔥 Q&A Section
    st.subheader("💬 Ask Questions from Document")

    query = st.text_input("Ask something...")

    if query:
        relevant_chunks = retrieve_chunks(query, model, index, chunks)
        answer = summarize_chunks(relevant_chunks)

        st.write("🧠 Answer:")
        st.write(answer)