import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

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

def generate_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, show_progress_bar=True)
    return model, embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def semantic_search(query, model, index, documents, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [(documents[i], distances[0][rank]) for rank, i in enumerate(indices[0])]

# Run full flow
pdf_path = "./aws-overview.pdf"  # Make sure the PDF is in the same directory
documents = prepare_pdf_chunks(pdf_path)
model, embeddings = generate_embeddings(documents)
index = build_faiss_index(np.array(embeddings))

# Example query
query = "What are the benefits of AWS security?"
results = semantic_search(query, model, index, documents)

# Show results
df = pd.DataFrame(results, columns=["Matching Text", "Distance"])
print(df)
