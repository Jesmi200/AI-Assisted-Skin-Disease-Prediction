FROM python:3.10-slim

# System deps needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects port 7860 by default; Cloud Run injects $PORT
# at runtime and expects the container to listen on it. app.py already reads
# os.environ.get("PORT", 7860), so this works for both.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
