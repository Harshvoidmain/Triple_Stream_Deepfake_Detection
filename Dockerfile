# Use official PyTorch runtime image with CUDA 12.1 support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies (needed for OpenCV headless support, though it's headless so mostly minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application codebase
COPY . .

# Expose the API port
EXPOSE 5000

# Start the Flask app wrapping it in Gunicorn for production
# Workers=1 because we're doing heavy GPU inference per process. Threads=4 handles concurrent IO bound wait
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
