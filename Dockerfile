FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend static files
COPY frontend /app/frontend

# Copy the backend code
COPY backend /app/backend

# Copy the models (we use .dockerignore or Git LFS to manage sizes)
COPY models /app/models

# Expose the Hugging Face standard port
EXPOSE 7860

# Run the FastAPI server on 0.0.0.0:7860 (required by Hugging Face Spaces)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
