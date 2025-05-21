import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

### --- PHASE 1: Extract and Prepare Text from PDF ---

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [page.get_text().strip() for page in doc if page.get_text().strip()]
    return pages

def clean_text(text):
    return ' '.join(text.replace('\n', ' ').split())

def chunk_text(text, max_words=150):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def prepare_pdf_chunks(pdf_path):
    raw_pages = extract_text_from_pdf(pdf_path)
    all_chunks = []
    for page in raw_pages:
        cleaned = clean_text(page)
        chunks = chunk_text(cleaned)
        all_chunks.extend(chunks)
    return all_chunks

### --- PHASE 2: Generate Embeddings ---

def generate_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, show_progress_bar=True)
    return model, embeddings

### --- PHASE 3: Build FAISS Index ---

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

### --- PHASE 4: Semantic Search Function ---

def semantic_search(query, model, index, documents, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [(documents[i], distances[0][rank]) for rank, i in enumerate(indices[0])]

### --- PHASE 5: Streamlit UI ---

def run_streamlit_ui(model, index, documents):
    st.title("📘 AWS PDF Semantic Search")
    user_query = st.text_input("Enter your question about AWS:")
    
    if user_query:
        results = semantic_search(user_query, model, index, documents)
        st.markdown("### 🔍 Top Results")
        for idx, (text, score) in enumerate(results):
            st.markdown(f"**{idx+1}. (Score: {score:.2f})**")
            st.markdown(f"> {text}\n")

### --- MAIN EXECUTION ---

def main():
    pdf_path = "./aws-overview.pdf"  # your uploaded PDF
    st.sidebar.write("Loading and processing PDF...")
    
    # Prepare text chunks
    documents = prepare_pdf_chunks(pdf_path)
    
    # Generate embeddings
    model, embeddings = generate_embeddings(documents)
    
    # Build FAISS index
    index = build_faiss_index(np.array(embeddings))
    
    # Run UI
    run_streamlit_ui(model, index, documents)

if __name__ == "__main__":
    main()
