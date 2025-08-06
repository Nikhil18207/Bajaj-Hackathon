from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob

INPUT_DIR = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def chunk_text_files():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for file_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt"))):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print(f"[!] Skipping empty file: {file_path}")
            continue

        chunks = splitter.split_text(text)
        print(f"[✓] {file_path} → {len(chunks)} chunks")
        
        all_chunks.extend([(file_path, idx, chunk) for idx, chunk in enumerate(chunks)])

    return all_chunks  # [(file_path, chunk_idx, chunk_text)]

if __name__ == "__main__":
    chunks = chunk_text_files()
    print(f"\n[✓] Total Chunks: {len(chunks)}")
