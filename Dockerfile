FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Render sets $PORT automatically, default to 10000 for local
EXPOSE 10000

# Start with gunicorn
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} app:app
