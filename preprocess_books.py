import os
import json
from PyPDF2 import PdfReader

def extract_text_from_pdf_path(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

def split_text(text: str, max_len: int = 3000) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_len:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def preprocess_and_save_pdf(book_path: str, cache_path: str):
    print(f"Processing {book_path} ...")
    text = extract_text_from_pdf_path(book_path)
    chunks = split_text(text)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"✅ Cached: {cache_path}")

def walk_and_preprocess_books(root_dir="books", cache_dir="cache"):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                parts = full_path.split(os.sep)
                # Expected path: books/board/CBSE/english/class_10/math.pdf
                if len(parts) >= 6:
                    _, _, board, language, class_folder, subject_pdf = parts[-6:]
                    class_level = class_folder.split("_")[1]
                    subject = subject_pdf.replace(".pdf", "")
                    cache_path = os.path.join(
                        cache_dir,
                        f"{board}_{language}_class{class_level}_{subject}.json"
                    )
                    preprocess_and_save_pdf(full_path, cache_path)

if __name__ == "__main__":
    walk_and_preprocess_books()
