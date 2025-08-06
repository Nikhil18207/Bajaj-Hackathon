import fitz  # PyMuPDF
import os

RAW_PDF_DIR = "data/raw_pdfs"
OUTPUT_TEXT_DIR = "docs"

os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def convert_all_pdfs():
    for filename in os.listdir(RAW_PDF_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(RAW_PDF_DIR, filename)
            try:
                text = extract_text_from_pdf(pdf_path)
                if not text.strip():
                    print(f"[!] Skipped empty PDF: {filename}")
                    continue

                output_filename = os.path.splitext(filename)[0] + ".txt"
                output_path = os.path.join(OUTPUT_TEXT_DIR, output_filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[✓] Saved: {output_path}")

            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")


if __name__ == "__main__":
    convert_all_pdfs()
