import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import faiss
import numpy as np
import pickle
from embeddings.embedder import BGEEmbedder

FAISS_INDEX_PATH = "retrieval/my_index.faiss"
TEXT_MAPPING_PATH = "retrieval/text_mapping.pkl"

class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def build_and_save_index(self, embedded_data):
        print("[*] Building FAISS index...")
        embeddings = [item['embedding'].cpu().numpy() for item in embedded_data]
        embeddings = np.array(embeddings).astype('float32')
        
        
        if embeddings.shape[0] == 0:
            print("[!] No embeddings found to build index.")
            return

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"[✓] Index built with {self.index.ntotal} vectors.")
        
        
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        
        
        self.texts = [item for item in embedded_data]
        with open(TEXT_MAPPING_PATH, "wb") as f:
            pickle.dump(self.texts, f)
            
        print(f"[✓] Index and text mapping saved.")

    def load_index(self):
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(TEXT_MAPPING_PATH, "rb") as f:
                self.texts = pickle.load(f)
            print(f"[✓] FAISS index and text mapping loaded.")
            return True
        except (IOError, FileNotFoundError, RuntimeError):
            print("[!] No existing FAISS index found. Please build the index first.")
            return False

if __name__ == "__main__":
    faiss_store = FaissVectorStore()
    

    if faiss_store.load_index():
        print("Index already exists. No need to rebuild.")
    else:
        
        embedder = BGEEmbedder()
        embedded_data = embedder.embed_chunks()
        
        if embedded_data:
            faiss_store.build_and_save_index(embedded_data)