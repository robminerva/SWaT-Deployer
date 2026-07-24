FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We'll map the codebase into /app at runtime
CMD ["python", "backend/xgboost_trainer/train_xgboost.py"]
