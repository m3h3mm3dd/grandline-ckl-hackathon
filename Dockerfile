FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads data/mcap

WORKDIR /app/backend

EXPOSE 8000

ENV PRELOAD_DATA_DIR=/app/data/mcap
ENV BND_PATH=/app/data/yas_marina_bnd.json
ENV UPLOAD_DIR=/app/uploads

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
