# 🔍 AWS Semantic Search (Vector DB Project)

This project implements a **semantic search engine** using AWS documentation as the source material. It extracts text from a PDF, embeds it using `sentence-transformers`, stores vectors in a `FAISS` index, and serves a searchable interface via `Streamlit`.

---

## 📚 Overview

- 📄 **Input**: `aws-overview.pdf` (AWS whitepaper)
- 🤖 **Embedding Model**: `all-MiniLM-L6-v2` from HuggingFace `sentence-transformers`
- 📦 **Vector Index**: `FAISS` for efficient similarity search
- 💻 **Frontend**: `Streamlit` web app
- 🔍 **Search Method**: Query text is embedded and compared to document chunks using cosine similarity

---

## 🚀 Features

- Extract and preprocess text from PDFs
- Generate embeddings for each chunk
- Search semantically across content using a FAISS index
- View ranked results in an interactive Streamlit UI

---

## 🛠️ Tech Stack

| Layer            | Tool/Library                |
|------------------|-----------------------------|
| Vector DB        | FAISS                       |
| Embeddings       | SentenceTransformers        |
| Text Extraction  | PyMuPDF (`fitz`)            |
| Web Interface    | Streamlit                   |
| Language         | Python                      |

---

## 🧪 How to Run

1. Clone the repository

```bash
git clone https://github.com/yuno-gen/aws-vector-search.git
cd aws-vector-search
