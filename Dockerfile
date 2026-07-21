# ---- cs-intel API + pipeline image ---------------------------------------
FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by xgboost's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY common ./common
COPY datagen ./datagen
COPY churn ./churn
COPY intents ./intents
COPY causal ./causal
COPY agent ./agent
COPY api ./api
COPY scripts ./scripts
COPY pyproject.toml ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CS_INTEL_DATA_DIR=/app/data \
    CS_INTEL_ARTIFACTS_DIR=/app/artifacts \
    CS_INTEL_MEMORY_DIR=/app/agent/memory_store \
    CS_INTEL_LLM_PROVIDER=mock

RUN mkdir -p /app/data /app/artifacts /app/agent/memory_store

EXPOSE 8000
CMD ["sh", "scripts/entrypoint.sh"]
