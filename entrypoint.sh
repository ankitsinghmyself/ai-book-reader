#!/bin/bash

# 🔧 Pre-start commands
echo "Running setup..."
python preprocess_books.py
python build_all_vector_indexes.py

# 🚀 Start FastAPI app
exec uvicorn main:app --host 0.0.0.0 --port $PORT
