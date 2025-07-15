import os
import json
import pickle
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss

# Load chunks grouped by (board, language, class, subject)
def group_chunks_by_metadata(cache_dir="cache"):
    grouped = defaultdict(list)
    pattern = re.compile(r"(.+?)_(.+?)_class(\d+?)_(.+?)_.*\.json")

    for file in os.listdir(cache_dir):
        match = pattern.match(file)
        if match:
            board, language, class_level, subject = match.groups()
            key = (board, language, class_level, subject)
            with open(os.path.join(cache_dir, file), "r", encoding="utf-8") as f:
                chunks = json.load(f)
                grouped[key].extend(chunks)
    return grouped

# Build and save FAISS index
def build_index(chunks, index_path, chunks_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Index saved: {index_path}")
    print(f"✅ Chunks saved: {chunks_path}")

if __name__ == "__main__":
    cache_dir = "cache"
    vector_dir = "vector_index"
    os.makedirs(vector_dir, exist_ok=True)

    grouped_chunks = group_chunks_by_metadata(cache_dir)

    for (board, language, class_level, subject), chunks in grouped_chunks.items():
        print(f"\n📚 Building index for: {board} | {language} | Class {class_level} | {subject} ({len(chunks)} chunks)")

        prefix = f"{board}_{language}_class{class_level}_{subject}"
        index_path = os.path.join(vector_dir, f"{prefix}.index")
        chunks_path = os.path.join(vector_dir, f"{prefix}_chunks.pkl")

        build_index(chunks, index_path, chunks_path)

    print("\n✅ All vector indexes built.")
