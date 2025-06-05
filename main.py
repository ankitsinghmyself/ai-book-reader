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

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ Request body model
class QuestionRequest(BaseModel):
    board: str = "UPMSP"
    classLevel: str = "12"
    language: str = "hindi"  # From dropdown
    subject: str = "math"
    question: str

# ✅ Extract text from PDF file at given path
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
def split_text(text, max_len=3000):
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

# ✅ Language-based system prompt generator
def get_system_prompt(language: str, class_level: str) -> str:
    language = language.lower()
    if language == "hindi":
        return f"आप कक्षा {class_level} के एक विशेषज्ञ शिक्षक हैं। कृपया पाठ्यपुस्तक की सामग्री का उपयोग करके प्रश्न का उत्तर हिंदी में दीजिए।"
    elif language == "marathi":
        return f"तुम्ही इयत्ता {class_level} साठी एक तज्ज्ञ शिक्षक आहात. कृपया पाठ्यपुस्तकातील मजकुराचा वापर करून मराठीत उत्तर द्या."
    elif language == "english":
        return f"You are an expert teacher for class {class_level}. Use the textbook content to answer the question in English."
    else:
        return f"You are a knowledgeable teacher. Answer the question using the textbook content."

# ✅ Main API endpoint
@app.post("/ask")
async def ask_ai(data: QuestionRequest):
    board = data.board
    class_level = data.classLevel
    language = data.language
    subject = data.subject
    question = data.question

    # 📚 Book path like: books/board/UPMSP/hindi/class_10/math.pdf
    book_path = f"books/board/{board}/{language}/class_{class_level}/{subject}.pdf"

    if not os.path.exists(book_path):
        return {"error": f"Book not found at: {book_path}"}

    # 🧠 Extract and split book content
    text = extract_text_from_pdf_path(book_path)
    chunks = split_text(text)

    # 🧑‍🏫 Dynamic prompt based on selected language
    messages = [
        {"role": "system", "content": get_system_prompt(language, class_level)},
        {"role": "user", "content": question},
        {"role": "user", "content": chunks[0]}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
