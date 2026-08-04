#!/usr/bin/env bash
#
# Bring a bare Ubuntu 24.04 EC2 instance to a running AutoDS stack. Intended as
# EC2 user-data, but written to be safe to run by hand over SSH as well --
# every step is idempotent, so re-running after a failure is fine.
#
# NOT YET RUN against a real instance -- see docs/DEPLOYMENT.md for which steps
# are verified and which are reasoned.
#
# It does NOT write the secrets. It creates /etc/autods/autods.env with the
# right ownership and mode and stops if the values are absent, because baking
# secrets into user-data would put them in the instance metadata service, where
# anything that can reach 169.254.169.254 can read them back.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/VatsalTheCoder/autods.git}"
APP_DIR="${APP_DIR:-/opt/autods}"
ENV_FILE="/etc/autods/autods.env"

log() { echo "[bootstrap] $*"; }

log "Installing Docker Engine and the compose plugin"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl git

# Docker's own repository rather than Ubuntu's docker.io package: the compose
# *plugin* (`docker compose`, not `docker-compose`) is only packaged there, and
# every command in this repo assumes the plugin form.
install -m 0755 -d /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/docker.asc ]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

# So the stack comes back after a reboot. `restart: unless-stopped` in the
# compose file only helps if the daemon itself starts.
systemctl enable --now docker

log "Fetching the application to ${APP_DIR}"
if [ -d "${APP_DIR}/.git" ]; then
  git -C "${APP_DIR}" fetch --quiet origin
  git -C "${APP_DIR}" reset --hard --quiet origin/main
else
  git clone --quiet --depth 1 "${REPO_URL}" "${APP_DIR}"
fi

log "Preparing ${ENV_FILE}"
install -d -m 0700 /etc/autods
if [ ! -f "${ENV_FILE}" ]; then
  cat > "${ENV_FILE}" <<'TEMPLATE'
# Filled in by hand on the instance. Root-owned, mode 600, never in git.
#
# POSTGRES_PASSWORD  the database password. Needed here rather than in Secrets
#                    Manager because Postgres must be up before anything can
#                    authenticate to AWS to read a secret -- the bootstrap
#                    chicken and egg. Generate with `openssl rand -base64 24`.
# AWS_SECRETS_ID     name of the Secrets Manager secret holding google_api_key.
# AUTODS_ACCESS_TOKEN  the shared secret the Streamlit gate demands.
POSTGRES_PASSWORD=
AWS_SECRETS_ID=autods/production
AUTODS_ACCESS_TOKEN=
AWS_REGION=eu-west-2
TEMPLATE
  chmod 600 "${ENV_FILE}"
  log "Wrote a template. Fill it in, then re-run this script."
  exit 1
fi
chmod 600 "${ENV_FILE}"

# shellcheck disable=SC1090
set -a; . "${ENV_FILE}"; set +a
: "${POSTGRES_PASSWORD:?empty in ${ENV_FILE}}"
: "${AWS_SECRETS_ID:?empty in ${ENV_FILE}}"
: "${AUTODS_ACCESS_TOKEN:?empty in ${ENV_FILE}}"

cd "${APP_DIR}"

log "Building the image (several minutes: scientific wheels and WeasyPrint)"
docker compose -f docker-compose.prod.yml build

log "Starting Postgres and Redis"
docker compose -f docker-compose.prod.yml up -d postgres redis

# Migrations before the app, not alongside it. The api container would otherwise
# race the schema it queries, and `depends_on: service_healthy` cannot express
# "and the tables exist".
log "Applying migrations"
docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head

log "Starting the application"
docker compose -f docker-compose.prod.yml up -d

log "Waiting for the UI to report healthy"
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    log "up"
    exit 0
  fi
  sleep 5
done

log "UI did not become healthy within five minutes; check:"
log "  docker compose -f ${APP_DIR}/docker-compose.prod.yml logs --tail=50"
exit 1
