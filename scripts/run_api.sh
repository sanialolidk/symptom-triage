#!/bin/bash
# Port 8001 — climate-signal already uses 8000
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
uvicorn api.main:app --reload --host 127.0.0.1 --port 8001