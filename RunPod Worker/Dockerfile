# Example base image; in RunPod UI you can choose an image that matches this.
FROM runpod/pytorch:2.4.0-py3.10-cuda12.1-runtime

WORKDIR /app

# Copy code
COPY handler.py /app/handler.py
COPY requirements.txt /app/requirements.txt

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# RunPod will call "python -m handler" or similar via runpod.serverless.start
CMD ["python", "-u", "handler.py"]
