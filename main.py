from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import json
import pickle
import faiss
from uuid import uuid4
from typing import Optional
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
    language: str  # Output language (english/hindi)
    subject: str
    question: str
    session_id: Optional[str] = None

# In-memory session context
session_context = {}

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Semantic search
def search_semantic_chunks(board, language, class_level, subject, question, top_k=20):
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

# Prompt builder
def build_system_prompt(language: str, class_level: str, book_context: str) -> str:
    language = language.lower()
    if language == "hindi":
        return (
            f"आप कक्षा {class_level} के एक अनुभवी बोर्ड परीक्षा शिक्षक हैं। "
            f"पाठ्यपुस्तक की सामग्री का उपयोग करते हुए प्रश्न का उत्तर बोर्ड परीक्षा में दिए जाने वाले उत्तर के अनुसार "
            f"स्पष्ट, सटीक और क्रमबद्ध रूप में हिंदी में लिखिए।\n\n"
            f"यह पाठ्यपुस्तक की सामग्री है:\n{book_context}\n"
            f"अब छात्र का प्रश्न है:"
        )
    else:
        return (
            f"You are an experienced board exam teacher for class {class_level}. "
            f"Using the textbook content, write the answer in clear, concise, and structured English, "
            f"as expected in a board examination.\n\n"
            f"Here is the textbook content:\n{book_context}\n"
            f"Now the student's question is:"
        )



@app.post("/ask")
async def ask_ai(data: QuestionRequest):
    board = data.board
    class_level = data.classLevel
    language = data.language.lower()
    subject = data.subject
    question = data.question.strip()
    session_id = data.session_id or str(uuid4())

    follow_ups = [
        "i didn't understand", "explain again", "what does it mean", "can you elaborate",
        "और बताओ", "फिर से समझाओ", "next", "more"
    ]
    is_followup = question.lower() in follow_ups

    if is_followup and session_id in session_context:
        last_question = session_context[session_id]["question"]
        book_context = session_context[session_id]["book_context"]

        prompt = (
            build_system_prompt(language, class_level, book_context)
            + f"\n\nStudent's original question: {last_question}\nPlease explain again in simpler terms."
        )

    else:
        chunks = search_semantic_chunks(board, language, class_level, subject, question, top_k=30)
        fallback = False

        if not chunks or all(len(c.strip()) < 20 for c in chunks):
            fallback = True
            prompt = (
                f"आप एक अनुभवी कक्षा {class_level} के गणित शिक्षक हैं। छात्र ने यह प्रश्न पूछा है:\n\n"
                f"❓ \"{question}\"\n\n"
                f"यह जानकारी पाठ्यपुस्तक में स्पष्ट रूप से उपलब्ध नहीं है। कृपया अपना सामान्य ज्ञान और अनुभव उपयोग करके उत्तर दें, "
                f"परंतु उत्तर कक्षा {class_level} की गणित स्तर के अनुसार हो। उत्तर क्रमबद्ध और परीक्षा-योग्य होना चाहिए।"
                if language == "hindi"
                else
                f"You are an experienced class {class_level} mathematics teacher.\n\n"
                f"The student asked:\n❓ \"{question}\"\n\n"
                f"This content is not clearly present in the textbook. Please use your general knowledge to answer, "
                f"but keep the answer aligned with class {class_level} syllabus and in exam-appropriate structured format."
            )
            book_context = ""  # So it's stored even for follow-up
        else:
            book_context = "\n".join(chunks)
            is_summary = any(kw in question.lower() for kw in ["important topic", "key point", "main point", "summary"])
            fallback = False

            if is_summary:
                prompt = (
                    f"आप कक्षा {class_level} के एक अनुभवी बोर्ड परीक्षा शिक्षक हैं। "
                    f"नीचे दी गई पाठ्यपुस्तक सामग्री का विश्लेषण करके महत्वपूर्ण विषयों की सूची तैयार कीजिए। बिंदुवार उत्तर दें:\n\n{book_context}"
                    if language == "hindi"
                    else
                    f"You are an experienced class {class_level} teacher. Analyze the textbook content below and list key topics or summary points:\n\n{book_context}"
                )
            else:
                prompt = (
                    build_system_prompt(language, class_level, book_context) + f"\n\n{question}"
                )

        # Save session context
        session_context[session_id] = {
            "question": question,
            "book_context": book_context
        }

    # Call Gemini API
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return {
            "answer": response.text.strip(),
            "session_id": session_id,
            "source": "AI" if fallback else "book"
        }
    except Exception as e:
        return {"error": str(e)}
