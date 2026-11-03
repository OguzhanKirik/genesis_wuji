#!/bin/bash
# Stop both containers (the images and the logs stay).
set -euo pipefail
cd "$(dirname "$0")"
docker compose down
