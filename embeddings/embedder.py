from sentence_transformers import SentenceTransformer
import torch
from parsing.chunker import chunk_text_files
import os

class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[✓] Loaded model: {model_name} on {self.device}")

    def embed_text(self, text: str):
        return self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    def embed_chunks(self):
        chunks = chunk_text_files()
        embedded_data = []

        for file_path, chunk_id, chunk_text in chunks:
            if not chunk_text.strip():
                continue
            embedding = self.embed_text(chunk_text)
            embedded_data.append({
                "source": file_path,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "embedding": embedding
            })
            print(f"[✓] Embedded chunk {chunk_id} from {os.path.basename(file_path)}")

        return embedded_data

if __name__ == "__main__":
    embedder = BGEEmbedder()
    results = embedder.embed_chunks()
    print(f"\n[✓] Embedded {len(results)} chunks total.")
