# Use NVIDIA CUDA base image with cuDNN support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Add metadata labels
LABEL maintainer="Mtailor ML Team" \
      description="Image Classification API using ONNX model for Cerebrium deployment" \
      version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10 \
    PORT=8000 \
    CEREBRIUM_MODEL_PATH=/app/model.onnx \
    CEREBRIUM_API_KEY="" \
    CEREBRIUM_WORKERS=1 \
    CEREBRIUM_TIMEOUT=60

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies and clean up pip cache
RUN pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip/*

# Copy application code and model
COPY src/ /app/src/
COPY model.onnx /app/

# Create a non-root user and set permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE ${PORT}

# Add healthcheck with Cerebrium-specific endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run the application with Cerebrium-specific settings
CMD ["/bin/bash", "-c", "python3 -m uvicorn src.app:app --host 0.0.0.0 --port ${PORT} --workers ${CEREBRIUM_WORKERS} --timeout-keep-alive ${CEREBRIUM_TIMEOUT}"] 