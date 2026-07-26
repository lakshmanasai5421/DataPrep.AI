# For Hugging Face Spaces (Docker SDK) or any container host.
# HF Spaces routes traffic to port 7860 by default.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Writable dirs (HF/most container FS are read-only except the app dir + /tmp)
RUN mkdir -p uploads static && chmod -R 777 uploads static

ENV PORT=7860
EXPOSE 7860

CMD gunicorn server:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT}
