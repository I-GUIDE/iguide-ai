FROM python:3.11-slim

# Install system dependencies (GDAL for geopandas/fiona used by the in-process
# MCP spatial tools the analysis agent loads)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    gnupg \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI (client only) so sandboxed code execution can shell out to the HOST
# Docker daemon (Docker-out-of-Docker). The host socket must be mounted at runtime;
# no daemon/containerd is installed here.
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# QGIS + PyQGIS bindings (Debian bookworm) so the analysis agent's QGIS tools work:
# pyqgis_render_map / pyqgis_layer_summary (via the worker) and the qgis_process CLI.
# The python3-qgis bindings install into the DISTRO python3 (/usr/bin/python3) — which is the
# interpreter the PyQGIS worker runs as (QGIS_PYTHON_BIN below), separate from the app's
# /usr/local/bin/python. --no-install-recommends drops GRASS/docs to keep the image smaller;
# vector (OGR) rendering needs only qgis-core, which python3-qgis pulls in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    qgis python3-qgis \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Headless matplotlib + package root on PYTHONPATH so the agent can import the
# in-process MCP tool modules (MCP_server/tools/*) and rag_pipeline/agent_runtime.
# GDAL include paths so fiona/rasterio build against the system GDAL.
ENV MPLBACKEND=Agg \
    PYTHONPATH=/app \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    # PyQGIS worker runs in the distro python3 (where python3-qgis installed the bindings),
    # not the app's /usr/local/bin/python. Offscreen QPA = headless rendering without an X server.
    QGIS_PYTHON_BIN=/usr/bin/python3 \
    QGIS_PREFIX_PATH=/usr \
    QT_QPA_PLATFORM=offscreen

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# py-spy reads the Python stacks of a RUNNING process from outside it, without the target's
# cooperation. deploy/agent-watchdog.sh runs it against this container before restarting one
# that has gone unhealthy, so a hang leaves behind the stack every thread was sitting on
# rather than just the fact that it happened. Its own layer: a few MB, and it must not
# invalidate the requirements layer above.
RUN pip install --no-cache-dir py-spy

# Install spaCy model for NLP processing
RUN python -m spacy download en_core_web_sm

# Copy application packages. MCP_server is required because the agent loads MCP
# tools in-process (agent_runtime.langchain_mcp_tools imports MCP_server/tools/*).
COPY rag_pipeline/ ./rag_pipeline/
COPY agent_runtime/ ./agent_runtime/
COPY api/ ./api/
COPY MCP_server/ ./MCP_server/
# Agent knowledge-base ingestion package — the search/code peers' agent_kb_search and
# get_kb_block tools import `extractors` (kb_store, indices) at call time.
COPY extractors/ ./extractors/

# Pre-download SentenceTransformer model so the container doesn't need
# outbound network access to HuggingFace on first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create writable upload directory and non-root user
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/agent_chat_files/uploads \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5002/health', timeout=5).raise_for_status()" || exit 1

# Run with gunicorn. Worker COUNT is intentionally NOT pinned here: gunicorn reads it from the
# WEB_CONCURRENCY env var (or defaults to 1). Keep it at 1 unless persistent memory (OpenSearch)
# is enabled — session memory + the LangGraph checkpointer are process-local, so >1 worker
# without shared/persistent memory breaks multi-turn continuity. Threads are I/O concurrency
# within each worker (they share process memory, so they don't affect that contract).
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--threads", "4", "--timeout", "600", "--access-logfile", "-", "--error-logfile", "-", "api.server:app"]
