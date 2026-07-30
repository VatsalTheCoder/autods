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

# curl is used by the compose healthchecks. libgomp1 is LightGBM's OpenMP
# runtime (Section 7): its wheel links against it but does not bundle it, so
# without this the import fails at container start with a bare
# "libgomp.so.1: cannot open shared object file".
#
# The pango/cairo/gdk-pixbuf trio is WeasyPrint's rendering stack (Section 9).
# WeasyPrint is pure Python but binds to these at runtime, so without them the
# *import* succeeds and the first PDF render fails -- which is the worst place
# to find out, since it is at the end of a pipeline run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz-subset0 \
        libgdk-pixbuf-2.0-0 \
        libffi-dev \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Dependencies are installed before the source is copied. Docker caches layers,
# so editing a Python file re-runs only the final COPY rather than a full
# reinstall -- the difference between a 2-second and a 90-second rebuild.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install ".[ui,dev]"

COPY app/ ./app/
COPY ui/ ./ui/
# Streamlit reads .streamlit/config.toml relative to the working directory, so
# without this the built image falls back to the stock theme and a deployed UI
# looks nothing like the one developed against.
COPY .streamlit/ ./.streamlit/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Run as a non-root user. If the container is ever compromised, the attacker
# does not land as root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /code
USER appuser

# Bake the embedding model into the image (Section 10). fastembed otherwise
# fetches it on first use, which would put a ~130 MB download inside the first
# question anyone asks -- and would make the container require network access at
# query time to answer from data it already has locally. Downloaded as appuser
# so the cache lands in the home directory the app actually runs from.
ENV FASTEMBED_CACHE_PATH=/home/appuser/.cache/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
