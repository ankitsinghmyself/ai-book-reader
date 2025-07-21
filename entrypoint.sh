#!/bin/bash

# 🔧 Pre-start commands
echo "Running setup..."
python preprocess_books.py
python build_all_vector_indexes.py

# 🚀 Start FastAPI app
python -m venv .venv
.\.venv\Scripts\activate  
uvicorn main:app --reload
     
