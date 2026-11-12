#!/usr/bin/env bash
# Starts the godsEye services on the Studio. Called by the Studio's on_start.sh and by `make up`.
set -euo pipefail
bash "$HOME/godsEye/infra/studio/bootstrap.sh"

PC=("$HOME/opt/bin/process-compose" --unix-socket "$HOME/data/pc.sock")
if "${PC[@]}" process list >/dev/null 2>&1; then
  echo "[start] already running"
else
  rm -f "$HOME/data/pc.sock"
  # The daemon inherits these fds; a file (not a pipe) lets the caller's SSH session exit.
  "${PC[@]}" up --detached --config "$HOME/godsEye/infra/studio/process-compose.yaml" \
    >>"$HOME/data/logs/process-compose-cli.log" 2>&1
  echo "[start] services started"
fi
