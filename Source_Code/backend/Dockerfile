FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# cache-bust: 2026-08-04-v4
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files to container root
COPY . .

# Run FastAPI using the dynamic port Railway assigns
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
