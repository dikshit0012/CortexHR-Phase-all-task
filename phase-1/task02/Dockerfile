# Reproducible environment for the CortexHR AI/ML project.
# Runs JupyterLab (default) or any script, e.g.:  docker compose run --rm audit
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies first so Docker caches this layer until requirements.txt changes
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Project code (data/ and reports/ are mounted as volumes at run time, see docker-compose.yml)
COPY . .

# Non-root user: a good habit even for a training project
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8888

# Default: start JupyterLab. Token is disabled for LOCAL use only - never expose this port publicly.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/app"]
