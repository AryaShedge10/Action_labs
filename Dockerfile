FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/samples ./data/samples

# Run as an unprivileged user; results (PHI) are written to a mounted volume only.
RUN useradd --create-home teamb && mkdir -p /app/data/store && chown -R teamb /app/data/store
USER teamb

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
