#!/usr/bin/env bash
# Stores the Anthropic API key in ~/data/secrets on the Studio. It takes precedence over Lightning's
# injected secret (services/search/llm.py), so the key can change without a Studio restart.
# Run from the Mac with `make set-anthropic-key`; the key is typed here and never echoed.
set -euo pipefail
read -rs -p "Anthropic API key: " key
echo
[[ $key == sk-ant-* ]] || { echo "doesn't look like an Anthropic key (sk-ant-...); not saved"; exit 1; }
umask 077
mkdir -p "$HOME/data/secrets"
printf %s "$key" >"$HOME/data/secrets/anthropic_api_key"
echo "saved (fingerprint $(printf %s "$key" | sha256sum | cut -c1-10))"
