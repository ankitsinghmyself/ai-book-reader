from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import openai
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# It's safer to set your OpenAI key as env variable and read it here
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    reader = PdfReader(pdf_file.file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

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

@app.post("/explain")
async def explain_pdf(
    file: UploadFile = File(...),
    question: str = Form(...),
    grade: str = Form("10th")  # User can specify 10th, 12th, etc.
):
    text = extract_text_from_pdf(file)
    if not text.strip():
        return {"error": "No text extracted from PDF"}

    chunks = split_text(text, max_len=3000)

    # System prompt explains the style
    system_message = (
        f"You are a friendly teacher. Answer the question as a student would in a {grade} grade exam. "
        "Use simple language, clear explanations, and provide examples where appropriate."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question+ " Please answer in a way that a student in grade " + grade + " write in exam."}
    ]

    context_text = "\n\n".join(chunks[:3])  # Include first few chunks of book content
    messages.append({"role": "user", "content": f"Here is the content from a book:\n{context_text}"})

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return {"error": str(e)}

    return {"answer": answer}
