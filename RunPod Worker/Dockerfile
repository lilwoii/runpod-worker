# Very simple, RunPod-friendly serverless worker Dockerfile

# Use a plain Python base image (no special CUDA image needed;
# RunPod attaches the GPU to this container)
FROM python:3.10-slim

# Work inside /app
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Start the worker
CMD ["python", "-u", "handler.py"]
