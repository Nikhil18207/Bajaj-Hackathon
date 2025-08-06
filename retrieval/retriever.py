import sys
import os

# Add the project's root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import faiss
import numpy as np
import pickle
from embeddings.embedder import BGEEmbedder
from retrieval.faiss_store import FAISS_INDEX_PATH, TEXT_MAPPING_PATH

class FaissRetriever:
    def __init__(self):
        self.embedder = BGEEmbedder()
        self.index = None
        self.texts = []

        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(TEXT_MAPPING_PATH, "rb") as f:
                self.texts = pickle.load(f)
            print(f"[✓] FAISS index and text mapping loaded for retrieval.")
        except (IOError, FileNotFoundError, RuntimeError) as e:
            print(f"[!] Error loading FAISS index: {e}")
            print("Please run retrieval/faiss_store.py to build the index first.")
            self.index = None

    def retrieve(self, query: str, k: int = 5):
        if self.index is None:
            return []

        # 1. Embed the query
        query_embedding = self.embedder.embed_text(query).cpu().numpy().astype('float32').reshape(1, -1)

        # 2. Search the FAISS index
        D, I = self.index.search(query_embedding, k)
        
        # 3. Retrieve the corresponding text chunks
        retrieved_chunks = []
        for i in range(k):
            chunk_index = I[0, i]
            if chunk_index >= len(self.texts):
                continue
            retrieved_chunks.append(self.texts[chunk_index])
        
        return retrieved_chunks

# Main script for testing
if __name__ == "__main__":
    retriever = FaissRetriever()
    if retriever.index:
        sample_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
        print(f"\n[*] Searching for query: '{sample_query}'")
        
        retrieved_results = retriever.retrieve(sample_query, k=5)
        
        print("\n[✓] Retrieved top 5 chunks:")
        for i, chunk in enumerate(retrieved_results):
            print(f"--- Chunk {i+1} (Source: {os.path.basename(chunk['source'])} | ID: {chunk['chunk_id']}) ---")
            print(chunk['text'][:200] + "...")
            print("-" * 20)