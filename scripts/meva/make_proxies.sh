#!/usr/bin/env bash
# Makes light viewing copies of MEVA clips: 720p (short side), 15 fps, ~0.9 Mbps, keyframe every 1 s,
# no B-frames (WebRTC-friendly). They are only for the browser; AI always reads the 1080p originals.
# The Mac <-> Studio tunnel tops out around 8 Mbps, so 1080p (~4 Mbps/camera) can't feed a camera wall.
#
# Usage: make_proxies.sh [G330 G339 ...]   (no args = every downloaded clip; existing proxies are skipped)
set -euo pipefail
RAW="$HOME/data/meva/raw"
OUT="$HOME/data/meva/proxy"
mkdir -p "$OUT"

clips() {
  if (($#)); then for cam in "$@"; do find "$RAW" -name "*.$cam.r13.avi"; done | sort
  else find "$RAW" -name "*.r13.avi" | sort; fi
}

clips "$@" | while read -r src; do
  dst="$OUT/$(basename "$src" .avi).mp4"
  [[ -s "$dst" ]] && continue
  echo "[proxy] $(basename "$dst")"
  ffmpeg -nostdin -v error -y -fflags +genpts -i "$src" \
    -vf "fps=15,scale=if(gt(iw\,ih)\,-2\,720):if(gt(iw\,ih)\,720\,-2)" \
    -c:v libx264 -preset veryfast -profile:v main -b:v 900k -maxrate 1200k -bufsize 2400k \
    -g 15 -keyint_min 15 -sc_threshold 0 -bf 0 -threads 4 -an -movflags +faststart \
    "$dst.tmp.mp4"
  mv "$dst.tmp.mp4" "$dst"
done
echo "[proxy] done"
