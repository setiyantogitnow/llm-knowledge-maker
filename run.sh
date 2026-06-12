#!/usr/bin/env bash

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Start the FastAPI app using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
