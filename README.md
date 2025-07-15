### https://ai-tutor-lilac.vercel.app/


### http://127.0.0.1:8000/docs

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

## Build vector indexes
python build_all_vector_indexes.py

## Run the server
uvicorn main:app --reload