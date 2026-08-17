FROM python:3.11-slim

# Install system dependencies for GDAL, PROJ, and spatial libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy application code.
#
# `extractors/` is NOT optional: minio_webhook.py's only real work is
# `from extractors.ingest import ingest_submission`, so without it every request fails with
# ModuleNotFoundError. The healthcheck hits /health, which imports none of this — so the
# container reported HEALTHY while being entirely non-functional. A liveness probe that cannot
# fail for the reason the service exists is not a probe.
#
# rag_pipeline is here for llm_utils (the publication extractor) and search.geo_shapes (dataset
# bbox reprojection); agent_runtime for file_store and element_resolver. Both are dependencies
# of `extractors` today — see the note in docs/DEPLOYMENT.md about which direction that should
# eventually point.
COPY metadata-extraction-server/ ./metadata-extraction-server/
COPY extractors/ ./extractors/
COPY rag_pipeline/ ./rag_pipeline/
COPY agent_runtime/ ./agent_runtime/

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health', timeout=5)" || exit 1

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--threads", "2", "--timeout", "300", "metadata-extraction-server.minio_webhook:app"]
