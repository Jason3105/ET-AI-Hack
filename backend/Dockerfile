# Lightweight runtime: Groq hosts all LLM and image analysis.
FROM node:20-slim AS node-builder
WORKDIR /bridge
COPY whatsapp-bridge/package.json whatsapp-bridge/package-lock.json* ./
RUN npm ci --omit=dev --no-audit --no-fund

FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PORT=10000 WHATSAPP_BRIDGE_URL=http://localhost:3001
WORKDIR /app

# Node is needed only for the existing Baileys WhatsApp sidecar. No FFmpeg,
# Tesseract, TensorFlow, PyTorch, OpenCV, or model-download build stage remains.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
COPY --from=node-builder /bridge/node_modules whatsapp-bridge/node_modules
RUN mkdir -p data_runtime whatsapp-bridge/.baileys_auth
EXPOSE 10000
CMD ["supervisord", "-c", "supervisord.conf"]
