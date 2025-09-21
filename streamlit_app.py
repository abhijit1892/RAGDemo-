"""Streamlit UI for Indian Constitution RAG Assistant"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.garph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="📜 Indian Constitution RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("📜 Indian Constitution RAG Assistant")
st.markdown("Ask questions about the **Constitution of India** and other legal documents.")

def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag(extra_pdfs=None, extra_urls=None):
    """Initialize RAG pipeline"""
    try:
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()

        # Collect sources
        sources = []
        sources.extend(Config.DEFAULT_URLS)
        sources.extend(Config.DEFAULT_PDFS)

        if extra_urls:
            sources.extend(extra_urls)
        if extra_pdfs:
            sources.extend(extra_pdfs)

        st.info(f"📚 Ingesting {len(sources)} sources...")
        documents = doc_processor.process_sources(sources)

        vector_store.create_vectorstore(documents)

        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return None, 0

def main():
    init_session_state()

    st.sidebar.header("⚙️ Options")
    st.sidebar.markdown("Add your own legal documents (PDFs) or URLs.")

    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )
    uploaded_paths = []
    if uploaded_files:
        Path("data").mkdir(exist_ok=True)
        for uf in uploaded_files:
            save_path = Path("data") / uf.name
            with open(save_path, "wb") as f:
                f.write(uf.read())
            uploaded_paths.append(str(save_path))

    # Custom URLs
    extra_urls = st.sidebar.text_area(
        "Add URLs (one per line)",
        ""
    ).splitlines()
    extra_urls = [u.strip() for u in extra_urls if u.strip()]

    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("🔄 Building knowledge base..."):
            rag_system, num_chunks = initialize_rag(
                extra_pdfs=uploaded_paths,
                extra_urls=extra_urls
            )
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ Ready! Indexed {num_chunks} text chunks.")

    st.markdown("---")

    # Ask question
    question = st.text_input("❓ Your Question", placeholder="e.g., What does Article 15 say?")
    if st.button("🔍 Search") and question:
        if st.session_state.rag_system:
            with st.spinner("⚖️ Consulting Constitution..."):
                start = time.time()
                result = st.session_state.rag_system.run(question)
                elapsed = time.time() - start

                # Extract results
                answer = getattr(result, "answer", None) or (result.get("answer") if isinstance(result, dict) else None)
                docs = getattr(result, "retrieved_docs", None) or (result.get("retrieved_docs") if isinstance(result, dict) else [])

                # Show
                st.subheader("💡 Answer")
                st.success(answer)

                # Show retrieved docs (clean display)
                if docs:
                    with st.expander("📄 Sources"):
                        for i, d in enumerate(docs, 1):
                            title = Path(d.metadata.get("source", "Document")).name
                            snippet = (d.page_content[:300] + "...") if d.page_content else "No content"
                            st.markdown(f"**Doc {i}: {title}**")
                            st.caption(snippet)

                st.caption(f"⏱️ Time: {elapsed:.2f}s")

                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "time": elapsed
                })

    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Recent Questions")
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"**Q:** {h['question']}")
            st.markdown(f"**A:** {h['answer'][:200]}...")
            st.caption(f"Time: {h['time']:.2f}s")

if __name__ == "__main__":
    main()
