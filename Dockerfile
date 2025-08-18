FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# GPU support if available
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV CUDA_VISIBLE_DEVICES=0

CMD ["gunicorn", "-w 4", "-k uvicorn.workers.UvicornWorker", "--timeout 120", "api:app"]