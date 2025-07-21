import os
import json
import pdfplumber
import re

def extract_text_from_pdf_path(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text_by_sentence(text: str, max_len: int = 3000) -> list[str]:
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_len:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def preprocess_and_save_pdf(book_path: str, cache_path: str):
    try:
        print(f"📘 Processing {book_path} ...")
        text = extract_text_from_pdf_path(book_path)
        chunks = split_text_by_sentence(text)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        print(f"✅ Cached: {cache_path}")
    except Exception as e:
        print(f"❌ Failed to process {book_path}: {e}")

def walk_and_preprocess_books(root_dir="books", cache_dir="cache"):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                parts = full_path.split(os.sep)

                # Expected: books/CBSE/english/class_10/math/pdf1.pdf
                if len(parts) >= 6:
                    _, board, language, class_folder, subject, pdf_file = parts[-6:]

                    class_level = class_folder.split("_")[1]
                    pdf_name = pdf_file.replace(".pdf", "")
                    subject = subject.lower()

                    cache_file = f"{board}_{language}_class{class_level}_{subject}_{pdf_name}.json"
                    cache_path = os.path.join(cache_dir, cache_file)

                    # ✅ Skip if already cached
                    if os.path.exists(cache_path):
                        print(f"⚠️ Skipping (already cached): {cache_path}")
                        continue

                    preprocess_and_save_pdf(full_path, cache_path)



if __name__ == "__main__":
    walk_and_preprocess_books()
