from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request model
class QuestionRequest(BaseModel):
    board: str
    classLevel: str
    language: str
    subject: str
    question: str

# Embedding model (shared instance)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔍 Semantic search function
def search_semantic_chunks(board, language, class_level, subject, question, top_k=3):
    prefix = f"{board}_{language}_class{class_level}_{subject}"
    index_path = f"vector_index/{prefix}.index"
    chunks_path = f"vector_index/{prefix}_chunks.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    q_embedding = embedding_model.encode([question])
    _, indices = index.search(q_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# 🧠 Prompt builder
def build_system_prompt(language: str, class_level: str, book_context: str) -> str:
    language = language.lower()
    if language == "hindi":
        return (
            f"आप कक्षा {class_level} के एक विशेषज्ञ शिक्षक हैं। "
            f"पाठ्यपुस्तक की सामग्री का उपयोग करके प्रश्न का उत्तर हिंदी में दीजिए।\n\n"
            f"यह पाठ्यपुस्तक की सामग्री है:\n{book_context}\n"
            f"अब छात्र का प्रश्न है:"
        )
    elif language == "marathi":
        return (
            f"तुम्ही इयत्ता {class_level} साठी एक तज्ज्ञ शिक्षक आहात. "
            f"पाठ्यपुस्तकातील मजकुराचा वापर करून प्रश्नाचे उत्तर मराठीत द्या.\n\n"
            f"ही पाठ्यपुस्तक सामग्री आहे:\n{book_context}\n"
            f"आता विद्यार्थ्याचा प्रश्न:"
        )
    else:
        return (
            f"You are an expert teacher for class {class_level}. "
            f"Use the textbook content to answer the question in English.\n\n"
            f"Here is the textbook content:\n{book_context}\n"
            f"Now the student's question is:"
        )

# 🚀 Main API endpoint
@app.post("/ask")
async def ask_ai(data: QuestionRequest):
    board = data.board
    class_level = data.classLevel
    language = data.language
    subject = data.subject
    question = data.question

    chunks = search_semantic_chunks(board, language, class_level, subject, question)
    if not chunks:
        return {"error": "Vector index not found. Please run the vector indexing script."}

    book_context = "\n".join(chunks)
    system_prompt = build_system_prompt(language, class_level, book_context)

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"{system_prompt}\n\n{question}")
        return {"answer": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}
