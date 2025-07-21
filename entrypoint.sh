#!/bin/bash

# 🔧 Pre-start commands
echo "Running setup..."


# 🚀 Start FastAPI app
python -m venv .venv
.\.venv\Scripts\activate  

python preprocess_books.py
python build_all_vector_indexes.py

uvicorn main:app --reload
     
