#!/usr/bin/env bash
# Hooks start.sh into the Studio's on_start.sh so services come back after every restart.
set -euo pipefail
HOOK="$HOME/.lightning_studio/on_start.sh"
MARK="# godsEye: start services"
grep -qF "$MARK" "$HOOK" || cat >>"$HOOK" <<EOF

$MARK
bash \$HOME/godsEye/infra/studio/start.sh >> \$HOME/data/logs/on_start.log 2>&1 &
EOF
echo "[on-start] hook installed"
