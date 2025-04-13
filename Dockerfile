# Use a minimal Alpine-based Python image for better security and smaller size
FROM python:3.13-alpine

WORKDIR /app

# Install system dependencies and necessary libraries with additional debugging
RUN apk update && apk add --no-cache \
    bash \
    git \
    build-base \
    gcc \
    g++ \
    libffi-dev \
    libssl3 \
    curl \
    cmake \
    python3-dev \
    libpq-dev \
    && apk clean

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files into container
COPY . .

# Default command to run the script
CMD ["python", "src/train.py"]
