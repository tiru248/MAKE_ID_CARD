# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files to container
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8080 for Railway
EXPOSE 8080

# Command to run the app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
