#!/usr/bin/env bash
# Replays downloaded MEVA clips as looping live cameras in MediaMTX (stream path: meva-<camera>).
# Each camera's clips are concatenated in time order, so the stream crosses file boundaries
# like a real recorder would. Video is passed through without re-encoding.
# MEVA AVIs carry no PTS and contain no B-frames (I/P only, keyframe every 2 s), so +genpts
# (PTS = DTS) gives correct timing without a transcode.
#
# Two variants per camera:
#   VARIANT=view (default): 720p proxies from make_proxies.sh -> path meva-<camera>   (for the browser)
#   VARIANT=hd:             1080p originals                     -> path meva-<camera>-hd (for AI, Studio-local)
#
# Usage: [VARIANT=hd] fake_live.sh start G330 G339 ... | stop | status
set -euo pipefail
VARIANT=${VARIANT:-view}
if [[ $VARIANT == hd ]]; then SRC="$HOME/data/meva/raw" EXT=avi SUFFIX=-hd; else SRC="$HOME/data/meva/proxy" EXT=mp4 SUFFIX=; fi
RUN="$HOME/data/run/fake_live"
mkdir -p "$RUN"

start_cam() {
  local cam=$1 name="meva-$1$SUFFIX"
  local list="$RUN/$name.txt"
  if [[ -f "$RUN/$name.pid" ]] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
    echo "$name already running"; return
  fi
  # Repeat the clips instead of -stream_loop: concat can't seek back in AVI ("Illegal seek"),
  # and a long list keeps timestamps monotonic. 48 passes of 30 min = 24 h.
  local clips; clips=$(find "$SRC" -name "*.$cam.r13.$EXT" | sort | sed "s/.*/file '&'/")
  : >"$list"
  [[ -n "$clips" ]] && for _ in $(seq 48); do echo "$clips" >>"$list"; done
  [[ -s "$list" ]] || { echo "no $VARIANT clips for $cam in $SRC"; return 1; }
  local pass; pass=$(cat "$HOME/data/secrets/mediamtx_publish_pass")
  # setsid: the restart loop and its ffmpeg share one process group, so stop can kill both.
  setsid nohup bash -c "while true; do
      ffmpeg -nostdin -hide_banner -loglevel warning -re -fflags +genpts \
        -f concat -safe 0 -i '$list' -map 0:v:0 -c copy \
        -f rtsp -rtsp_transport tcp 'rtsp://publisher:$pass@127.0.0.1:8554/$name'
      echo \"[fake-live] ffmpeg exited (\$?), restarting\"; sleep 2
    done" >"$RUN/$name.log" 2>&1 </dev/null &
  echo $! >"$RUN/$name.pid"
  echo "$name started ($(($(wc -l <"$list") / 48)) clips, looping)"
}

case "${1:-status}" in
  start)
    shift
    for cam in "$@"; do start_cam "$cam"; done
    ;;
  stop)
    for pid in "$RUN"/*.pid; do
      [[ -e "$pid" ]] || continue
      kill -- -"$(cat "$pid")" 2>/dev/null || true
      rm -f "$pid"
    done
    echo "stopped"
    ;;
  status)
    curl -s 127.0.0.1:9997/v3/paths/list | python3 -c "$(cat <<'PY'
import json, sys
for p in json.load(sys.stdin)["items"]:
    print(f"{p['name']:<12} ready={p['ready']}  readers={len(p.get('readers', []))}")
PY
)"
    ;;
esac
