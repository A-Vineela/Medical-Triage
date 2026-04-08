# Medical Triage Environment - OpenEnv Hackathon Submission
# HuggingFace Spaces compatible Dockerfile

FROM python:3.10

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ──────────────────────────────────────────────────────────────
COPY . .

# ── Environment variable defaults ────────────────────────────────────────────
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# Set OPENAI_API_KEY or HF_TOKEN at runtime via Space Secrets — never hardcode

# ── Resource constraints (2 vCPU / 8GB RAM — no heavy models loaded) ─────────
# All inference is via API calls, so no GPU/model weights needed

# ── Expose port for HuggingFace Spaces health check ─────────────────────────
EXPOSE 7860

# ── Default command: run the full benchmark ──────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]