FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps for llama-cpp GPU build + PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential cmake git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ ./app/
COPY handler.py .

# GLM-OCR model pulled at startup from HF via env var
ENV MODEL_PATH=/runpod-volume/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf
ENV GLM_OCR_PATH=/runpod-volume/models/GLM-OCR
ENV HF_HOME=/runpod-volume/models
ENV N_GPU_LAYERS=-1
ENV N_CTX=65536
ENV MAX_NEW_TOKENS=16384
ENV MAX_HEADER_CHARS=8000
ENV MAX_TXN_CHARS=48000
ENV CHUNK_SIZE=12000
ENV CHUNK_OVERLAP=500
ENV PAGE_DPI=150
ENV GLM_MAX_NEW_TOKENS=2048
ENV GLM_DEVICE=cuda
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["python", "-u", "handler.py"]