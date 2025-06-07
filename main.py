from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QuestionRequest(BaseModel):
    board: str
    classLevel: str
    language: str
    subject: str
    question: str

def load_chunks_from_cache(board, language, class_level, subject):
    cache_path = f"cache/{board}_{language}_class{class_level}_{subject}.json"
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

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

@app.post("/ask")
async def ask_ai(data: QuestionRequest):
    board = data.board
    class_level = data.classLevel
    language = data.language
    subject = data.subject
    question = data.question

    chunks = load_chunks_from_cache(board, language, class_level, subject)
    if not chunks:
        return {"error": "Book data not preprocessed yet. Please run preprocessing script."}

    # Use first 2 chunks for context (adjust if you want)
    book_context = "\n".join(chunks[:2])

    system_prompt = build_system_prompt(language, class_level, book_context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or your preferred model
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
