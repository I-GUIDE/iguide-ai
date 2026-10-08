FROM python:3.11-slim

# System deps: build tools + GDAL for geopandas/fiona/rasterio. libexpat1 is named on its own
# account: the rasterio and fiona wheels bundle their own GDAL, which links the system's
# libexpat.so.1, and python:3.11-slim does not ship it. The GDAL packages pull it in today;
# naming it keeps both imports working if those are ever dropped (see rag_pipeline/Dockerfile).
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gdal-bin \
    libgdal-dev \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Headless matplotlib and package root on PYTHONPATH so MCP tools can do
# `from rag_pipeline...` / `from agent_runtime...` without sys.path hacks.
# GDAL include paths so fiona/rasterio build against the system GDAL.
ENV MPLBACKEND=Agg \
    PYTHONPATH=/app \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements and install Python deps through the lock, as rag_pipeline/Dockerfile does.
# Both images install the same requirements.txt, and constraints.txt (the agent-api image's
# `pip freeze`) holds them to the same versions. Unpinned, every rebuild resolved fresh, so
# rebuilding one image alone (`up -d --build agent-api`) would leave the two on different stacks.
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy packages the tool modules import from
COPY MCP_server/ ./MCP_server/
COPY rag_pipeline/ ./rag_pipeline/
COPY agent_runtime/ ./agent_runtime/

# Create non-root user with writable upload dir
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/agent_chat_files/uploads \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=5)" || exit 1

CMD ["uvicorn", "MCP_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
