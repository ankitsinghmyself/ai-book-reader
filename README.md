### https://ai-tutor-lilac.vercel.app/


### http://127.0.0.1:8000/docs

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

## Build cache and vector indexes
python preprocess_books.py
python build_all_vector_indexes.py


## load books in json format
python preprocess_books.py;

## Run the server
uvicorn main:app --reload