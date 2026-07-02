import streamlit as st
import json
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Ask My PDF",
    page_icon="📄",
    layout="centered"
)

# App title
st.title("📄 Ask My PDF")
st.markdown("Upload a PDF and ask questions about its content using RAG.")

# --- Phase 5: Load BGE Embedding Model (cached so it loads only once) ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# --- Phase 2: PDF Loading ---
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_size_kb = len(uploaded_file.getvalue()) / 1024
    st.success("PDF uploaded successfully!")
    st.markdown(f"**File name:** {uploaded_file.name}")
    st.markdown(f"**File size:** {file_size_kb:.2f} KB")

    # --- Phase 3: Text Extraction ---
    reader = PdfReader(uploaded_file)
    extracted_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text

    st.markdown(f"**Extracted {len(extracted_text):,} characters** from {len(reader.pages)} page(s).")

    with st.expander("View Extracted Text"):
        st.text(extracted_text)

    # --- Phase 4: Chunking ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(extracted_text)

    st.markdown(f"**Created {len(chunks)} chunks** (chunk_size=500, chunk_overlap=50)")

    with st.expander("View First 3 Chunks"):
        for i, chunk in enumerate(chunks[:3]):
            st.markdown(f"**Chunk {i + 1}** ({len(chunk)} chars)")
            st.text(chunk)
            if i < 2 and i < len(chunks) - 1:
                st.divider()

    # --- Phase 5: Embedding Generation (BGE) ---
    embedding_model = load_embedding_model()

    with st.spinner("Generating embeddings for all chunks..."):
        embeddings = embedding_model.embed_documents(chunks)

    embedding_dim = len(embeddings[0])
    st.markdown(f"**Embedding dimension:** {embedding_dim}")
    st.markdown(f"**Generated {len(embeddings)} embeddings** for {len(chunks)} chunks.")

    with st.expander("View First Embedding (Preview)"):
        st.markdown(f"**Chunk 1 embedding** (first 5 values):")
        st.code(str(embeddings[0][:5]))

    # Save debug file for learning
    os.makedirs("debug", exist_ok=True)
    debug_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        debug_data.append({
            "chunk_index": i,
            "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
            "embedding": embedding[:10],  # Save first 10 values to keep file small
            "embedding_dimension": embedding_dim
        })

    with open("debug/embeddings.json", "w") as f:
        json.dump(debug_data, f, indent=2)

    st.success("Saved debug/embeddings.json for learning & inspection.")

    # --- Phase 6: Vector Database (ChromaDB) ---
    st.header("Vector Database (ChromaDB)")
    
    with st.spinner("Storing chunks in ChromaDB..."):
        # Generate metadata for each chunk
        metadatas = [
            {"source": uploaded_file.name, "chunk_index": i} 
            for i in range(len(chunks))
        ]
        
        # Load texts and metadatas into in-memory ChromaDB
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            metadatas=metadatas
        )

    st.success(f"Stored {len(chunks)} chunks in the vector database.")
    
    # Retrieve documents to verify collection count
    collection = vector_store.get()
    num_docs = len(collection["ids"])
    st.markdown(f"**Total documents in collection:** {num_docs}")

    # Display sample stored document for learning
    with st.expander("View Sample Stored Document (Debug)"):
        if num_docs > 0:
            sample_id = collection["ids"][0]
            sample_doc = collection["documents"][0]
            sample_meta = collection["metadatas"][0]
            
            st.markdown(f"**Document ID:** `{sample_id}`")
            st.markdown(f"**Metadata:**")
            st.json(sample_meta)
            st.markdown(f"**Content:**")
            st.text(sample_doc)
        else:
            st.warning("No documents found in collection.")
else:
    st.info("Please upload a PDF to get started.")

# --- Phase 7: User Question ---
st.divider()
st.header("💬 Ask a Question")
question = st.text_input(
    "Enter your question:",
    placeholder="Ask a question about your PDF...",
    disabled=uploaded_file is None
)

if question:
    st.info(f"**Question submitted:** {question}")

    # --- Phase 8: Question Embedding ---
    embedding_model = load_embedding_model()
    
    with st.spinner("Embedding your question..."):
        question_embedding = embedding_model.embed_query(question)
        
    with st.expander("View Question Embedding (Debug)"):
        st.markdown(f"**Question:** {question}")
        st.markdown(f"**Embedding dimension:** {len(question_embedding)}")
        st.markdown(f"**Embedding preview (first 5 values):**")
        st.code(str(question_embedding[:5]))

    # --- Phase 9: Retrieval ---
    st.subheader("🔍 Retrieval Results")
    
    # Ensure vector_store is in scope (initialized in upper block when uploaded_file is not None)
    if "vector_store" in locals():
        with st.spinner("Searching the vector database..."):
            retrieved_results = vector_store.similarity_search_with_score(question, k=3)
            
        # Map metadata chunk_index to ChromaDB IDs
        all_data = vector_store.get()
        id_map = {}
        if all_data and "ids" in all_data:
            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
                if meta and "chunk_index" in meta:
                    id_map[meta["chunk_index"]] = doc_id
                    
        st.success(f"Retrieved {len(retrieved_results)} relevant chunks from the database.")
        
        with st.expander("View Retrieved Chunks (Debug)", expanded=True):
            for idx, (doc, score) in enumerate(retrieved_results):
                chunk_idx = doc.metadata.get("chunk_index", "Unknown")
                doc_id = id_map.get(chunk_idx, "Unknown ID")
                
                # Dynamic keyword matching explanation
                words_in_common = [w for w in question.lower().split() if len(w) > 3 and w in doc.page_content.lower()]
                common_str = ", ".join([f"'{w}'" for w in words_in_common]) if words_in_common else "semantic context"
                
                explanation = (
                    f"This chunk was retrieved with a similarity distance score of {score:.4f}. "
                    f"It was selected because it contains keywords or semantic concepts related to your query "
                    f"(matching terms: {common_str})."
                )
                
                st.markdown(f"### 📄 Chunk {idx + 1}")
                st.markdown(f"**Document ID:** `{doc_id}`")
                st.markdown(f"**Similarity Score (L2 Distance):** `{score:.4f}` *(lower is closer)*")
                st.markdown(f"**Why this chunk was retrieved:** {explanation}")
                st.markdown(f"**Chunk Text:**")
                st.text(doc.page_content)
                if idx < len(retrieved_results) - 1:
                    st.divider()
                    
        # --- Phase 10: Answer Generation ---
        st.subheader("🤖 Answer Generation")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            st.error("GROQ_API_KEY is not configured in your .env file. Please add your actual key.")
        else:
            with st.spinner("Generating answer using Groq..."):
                try:
                    # Construct prompt context from retrieved chunks
                    context_text = "\n\n".join([f"[Source: Chunk {i+1}] {doc.page_content}" for i, (doc, _) in enumerate(retrieved_results)])
                    
                    messages = [
                        (
                            "system",
                            "You are a helpful assistant. Answer the user's question based ONLY on the provided context. "
                            "If the context does not contain the answer, say 'I don't have enough information to answer this question.' "
                            "Do not make up answers or use external knowledge."
                        ),
                        (
                            "user",
                            f"Context:\n{context_text}\n\nQuestion: {question}"
                        )
                    ]
                    
                    # Initialize LLM with temperature=0.0 for deterministic factual answers
                    llm = ChatGroq(
                        api_key=api_key,
                        model_name="llama-3.1-8b-instant",
                        temperature=0.0
                    )
                    
                    response = llm.invoke(messages)
                    
                    st.success("Answer generated successfully!")
                    st.write(response.content)
                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")
    else:
        st.warning("Vector store is not initialized. Please upload a PDF first.")
