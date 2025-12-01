# Simple, RunPod-friendly Dockerfile

FROM python:3.11.1-buster

# Work directory
WORKDIR /app

# Install deps early so they get cached
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy handler
COPY handler.py /app/handler.py

# Entry point – RunPod serverless will just run this container
CMD ["python", "-u", "/app/handler.py"]
