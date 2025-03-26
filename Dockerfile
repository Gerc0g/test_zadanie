FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs db

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]