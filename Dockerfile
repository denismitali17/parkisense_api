# Force a rock-solid Python 3.10 stable debian environment
FROM python:3.10-slim-bullseye

# Install system audio dependencies required by librosa and soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Establish our working directory inside the container image
WORKDIR /app

# Copy dependency files first to utilize Docker's build caching mechanisms
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel to safely unpack old binary structures
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python application dependencies safely from pre-built wheels
RUN pip install --no-cache-dir -r requirements.txt

# Copy all model files and Python source scripts into the working engine container
COPY . .

# Expose the internal network port standard used by Render
EXPOSE 10000

# Execute the FastAPI server through uvicorn bound to Render's required port configurations
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]