# Serving image: packages the TRAINED artifacts + API.
# Train first (make train-pipeline or train-pipeline-fast), then: make docker-build
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY config.yaml .
COPY src/ src/
COPY utils/ utils/
COPY api/ api/
COPY artifacts/models/ artifacts/models/
COPY artifacts/encode/ artifacts/encode/
COPY artifacts/scale/ artifacts/scale/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
