# One image, three roles. The api, worker and ui services all run this same
# image and differ only in their command -- so a dependency that works in one
# cannot mysteriously be missing in another.

FROM python:3.12-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the container
# - PYTHONUNBUFFERED: logs appear immediately instead of being buffered,
#   which matters a lot when you are watching `docker compose logs`
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# curl is used by the compose healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies are installed before the source is copied. Docker caches layers,
# so editing a Python file re-runs only the final COPY rather than a full
# reinstall -- the difference between a 2-second and a 90-second rebuild.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install ".[ui,dev]"

COPY app/ ./app/
COPY ui/ ./ui/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Run as a non-root user. If the container is ever compromised, the attacker
# does not land as root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /code
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
