## Testing purpose 
from embeddings.embedder import BGEEmbedder

embedder = BGEEmbedder()
embeddings = embedder.embed_docs("docs/")

print(f"\n[✓] Total documents embedded: {len(embeddings)}")
for path, emb in embeddings:
    print(f"{path}: {emb.shape}")
