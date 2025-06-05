from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ Request schema
class QuestionRequest(BaseModel):
    board: str
    classLevel: str
    language: str
    subject: str
    question: str

# ✅ Extract PDF text
def extract_text_from_pdf_path(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

# ✅ Split large text into chunks
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

# ✅ Build system prompt based on language
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
    else:  # Default English
        return (
            f"You are an expert teacher for class {class_level}. "
            f"Use the textbook content to answer the question in English.\n\n"
            f"Here is the textbook content:\n{book_context}\n"
            f"Now the student's question is:"
        )

# ✅ Main API endpoint
@app.post("/ask")
async def ask_ai(data: QuestionRequest):
    board = data.board
    class_level = data.classLevel
    language = data.language
    subject = data.subject
    question = data.question

    # 📘 PDF path
    book_path = f"books/board/{board}/{language}/class_{class_level}/{subject}.pdf"

    if not os.path.exists(book_path):
        return {"error": f"Book not found at: {book_path}"}

    # 🧠 Extract and chunk textbook
    text = extract_text_from_pdf_path(book_path)
    chunks = split_text(text)
    book_context = "\n".join(chunks[:2])  # you can increase chunk count if needed

    # 🧑‍🏫 System prompt
    system_prompt = build_system_prompt(language, class_level, book_context)

    # 💬 Messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4o
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
