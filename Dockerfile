# Use an official, lightweight Python runtime base image
FROM python:3.10-slim

# Install system audio dependencies required by librosa and soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /lib/lists/*

# Establish the working directory inside the container image
WORKDIR /app

# Copy dependency files first to utilize Docker's build caching mechanisms
COPY requirements.txt .

# Install Python application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all model files and Python source scripts into the working engine container
COPY . .

# Expose the internal network port standard used by Render
EXPOSE 10000

# Execute the FastAPI server through uvicorn bound to Render's required port configurations
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]