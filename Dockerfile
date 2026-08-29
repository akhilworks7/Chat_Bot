FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OCR and PDF handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch and Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create persistent runtime directories
RUN mkdir -p data/documents data/processed output

# Hugging Face Spaces standard port is 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
