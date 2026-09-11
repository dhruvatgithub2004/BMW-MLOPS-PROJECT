FROM python:3.13-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY main.py config.yaml ./
COPY src/ ./src/
COPY models/ ./models/

RUN useradd -m appuser && mkdir -p logs && chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces (Docker SDK) expects the app to listen on 7860.
EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "main:app"]
