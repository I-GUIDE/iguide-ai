FROM python:3.11-slim

# Install system dependencies for GDAL, PROJ, and spatial libraries. libexpat1 is named on its
# own account: the rasterio and fiona wheels (extract_metadata.py imports both) bundle their
# own GDAL, which links the system's libexpat.so.1, and python:3.11-slim does not ship it
# (see rag_pipeline/Dockerfile).
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy requirements and install Python dependencies through the lock, as rag_pipeline/Dockerfile
# does: this is the same root requirements.txt, and constraints.txt pins what it resolves to.
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy application code
COPY metadata-extraction-server/ ./metadata-extraction-server/
# `extractors/` is NOT optional: minio_webhook.py's only real work is
# `from extractors.ingest import ingest_submission`, so without it every request fails with
# ModuleNotFoundError while /health (which imports none of it) reports HEALTHY. rag_pipeline and
# agent_runtime are dependencies of `extractors` (llm_utils, geo_shapes, file_store).
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
