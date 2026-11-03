#!/bin/bash
# Pull the two images at the tags in .env. Needs one `docker login ghcr.io` with a GitHub token that has
# read:packages (the packages are private); the login is stored by docker and survives reboots.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "no .env: cp .env.example .env and set the tags and ROS_DOMAIN_ID" >&2; exit 2; }
set -a; source .env; set +a
if ! grep -q '"ghcr.io"' "${DOCKER_CONFIG:-$HOME/.docker}/config.json" 2>/dev/null; then
  echo "not logged in to ghcr.io: run  echo <token> | docker login ghcr.io -u <github-user> --password-stdin" >&2
  exit 3
fi
echo "pulling adam-moveit:${ADAM_IMAGE_TAG}${ARCH_SUFFIX:-} and genesis-rl-deploy:${APP_TAG}${ARCH_SUFFIX:-} ..."
docker compose pull
docker image ls --format '{{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}' | grep -E 'adam-moveit|genesis-rl-deploy'
