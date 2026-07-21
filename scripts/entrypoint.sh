#!/bin/sh
# Container entrypoint: build the pipeline artifacts (idempotent — skipped if a
# persisted volume already has them) then serve the API.
set -e

echo "[entrypoint] building pipeline artifacts (if missing)…"
python scripts/build_pipeline.py

echo "[entrypoint] starting API on :8000"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
