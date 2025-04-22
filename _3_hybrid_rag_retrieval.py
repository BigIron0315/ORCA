import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# === CONFIG ===
index_dir = Path("./rag_index")
top_k = 5
query_dir = Path("./interim_results/_2_query_gen")
output_dir = Path("./interim_results/_3_retrieved_chunks")
output_dir.mkdir(exist_ok=True)

# === LOAD HYBRID INDEX ===
print("📦 Loading index files...")
with open(index_dir / "documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open(index_dir / "tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

faiss_index = faiss.read_index(str(index_dir / "dense_index.faiss"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# === PROCESS EACH QUERY FILE ===
for query_file in sorted(query_dir.glob("LLM_generated_queries_*.txt")):
    with open(query_file, "r") as f:
        queries = [q.strip("- ").strip() for q in f.readlines() if q.strip()]

    print(f"\n📄 Loaded {len(queries)} queries from {query_file.name}\n")

    output_lines = []

    for query in queries:
        print(f"🔍 Processing query: {query}")
        output_lines.append(f"\n=== Query: {query} ===\n")

        # DENSE SEARCH
        query_dense = model.encode([query])
        D, I = faiss_index.search(np.array(query_dense), top_k)
        retrieved_dense = [documents[i] for i in I[0]]

        # SPARSE SEARCH
        sparse_query_vec = tfidf_vectorizer.transform([query])
        doc_texts = [doc.page_content for doc in documents]
        sparse_doc_matrix = tfidf_vectorizer.transform(doc_texts)
        sim_scores = cosine_similarity(sparse_query_vec, sparse_doc_matrix).flatten()
        top_sparse_idx = np.argsort(sim_scores)[-top_k:][::-1]
        retrieved_sparse = [documents[i] for i in top_sparse_idx]

        # HYBRID MERGE
        combined = {doc.page_content: doc.metadata.get("source", "") for doc in retrieved_dense + retrieved_sparse}

        for i, (chunk, source) in enumerate(combined.items(), 1):
            output_lines.append(f"--- Result #{i} ---\n")
            output_lines.append(chunk.strip()[:800] + "\n")
            output_lines.append(f"📌 Source: {source}\n")

    output_file = output_dir / f"retrieved_chunks_{query_file.stem.replace('LLM_generated_queries_', '')}.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Retrieved chunks saved to {output_file}")