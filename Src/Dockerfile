# Use Python 3.12 slim base image
FROM python:3.12-slim

# Install system dependencies (gfortran for SciPy + build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    gfortran \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the application
CMD ["python", "app.py"]