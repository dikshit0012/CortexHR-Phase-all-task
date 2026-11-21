# CortexHR API image. Build context = REPOSITORY ROOT:
#   docker build -f task17/Dockerfile -t cortexhr-api .
#   docker run -p 8000:8000 -e CORTEXHR_API_KEYS=change-me cortexhr-api
#
# Python 3.14.6 and exact library pins are required: the Task 8 anomaly artifacts refuse to load on any other version.
# uv installs that exact interpreter, so the image does not depend on which 3.14.x tag the base registry currently serves.
FROM python:3.12-slim-bookworm AS build
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir uv==0.11.32
COPY task17/requirements.txt /tmp/requirements.txt
RUN uv python install 3.14.6 && uv venv /opt/venv --python 3.14.6 && uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/opt/venv/bin:$PATH" PORT=8000 CORTEXHR_ENV=production
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv
RUN useradd --create-home --uid 10001 cortexhr
WORKDIR /app
COPY --chown=cortexhr:cortexhr . /app
USER cortexhr
WORKDIR /app/task17
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/v1/health' % os.environ.get('PORT','8000'), timeout=4)"
# One worker: models are loaded once per process (~210 MB RSS measured). Scale with instances, not workers, on small plans.
CMD ["sh", "-c", "exec uvicorn cortexhr_api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
