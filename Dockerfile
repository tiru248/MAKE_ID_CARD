FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# For render, set to port 80 if required
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:80 app:app"]
