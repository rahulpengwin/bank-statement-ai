FROM runpod/base:0.6.2-cuda12.1.0

WORKDIR /workspace

# System dependencies for PyMuPDF, Pillow, llama-cpp-python
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python --force-reinstall

# Copy application code
COPY app/ ./app/
COPY handler.py .

# Environment variables — defaults, overridden by RunPod endpoint config
ENV MODEL_PATH=/runpod-volume/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf
ENV GLM_OCR_PATH=/runpod-volume/models/GLM-OCR
ENV N_GPU_LAYERS=-1
ENV N_CTX=8192
ENV GLM_MAX_NEW_TOKENS=8192
ENV PAGE_DPI=200
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]