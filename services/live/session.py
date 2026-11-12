"""Live indexing sessions on the Studio.

    MediaMTX rtsp://127.0.0.1:8554/<camera> -> FFmpeg, 5 fps JPEGs -> websocket -> Modal GPU (Live)
    rows every ~1 s <- websocket <- Modal                         -> Postgres, searchable at once

A session's rows go into run "live-<start time>" under the stream's name as the camera. Replays and
verifier evidence come from MediaMTX's recordings of the same stream (mediamtx.yml). Frames are stamped
with the Studio's clock as they arrive, the same clock MediaMTX names its recordings by. When the GPU
falls behind, the newest frames are dropped rather than queued, so results stay seconds behind live. A
session ends when it is stopped, when the stream ends, or after MAX_HOURS.
"""
from __future__ import annotations

import base64
import collections
import json
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector
from websockets.sync.client import connect

URL = "wss://iamoriginal67--godseye-indexer-live-web.modal.run/ws/{camera}"
TOKEN = Path.home() / "data/secrets/live_token"
DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
FPS = 5
IN_FLIGHT = 10        # frames sent and not yet processed; more than this and new frames are dropped
MAX_HOURS = 2.0       # a forgotten session stops itself (the GPU bills while it runs)
ORPHAN_S = 30.0       # crops whose track never got a second point are dropped after this


def _ts(s: float) -> datetime:
    return datetime.fromtimestamp(s, tz=timezone.utc)


def _vec(b64: str) -> HalfVector:
    return HalfVector(np.frombuffer(base64.b64decode(b64), dtype=np.float16))


def _bytes(b64: str | None) -> bytes | None:
    return base64.b64decode(b64) if b64 else None


class Session:
    def __init__(self, camera: str, on_change=lambda: None):
        self.camera = camera
        self.run = f"live-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
        self.on_change = on_change
        self.state, self.error = "starting the GPU", None
        self.started = time.time()
        self.sent = self.processed = self.dropped = 0
        self.lag_s: float | None = None
        self.counts: collections.Counter = collections.Counter()
        self.track_ids: dict[int, int] = {}        # live track id -> tracks.id
        self.waiting: list[tuple[float, str, dict]] = []   # crops/parts whose track isn't in the database yet
        self.stopping = threading.Event()
        self.ffmpeg: subprocess.Popen | None = None
        self.thread = threading.Thread(target=self._run, name=f"live-{camera}", daemon=True)
        self.thread.start()

    def status(self) -> dict:
        return {"camera": self.camera, "run": self.run, "state": self.state, "error": self.error,
                "seconds": round(time.time() - self.started), "frames_sent": self.sent, "frames_indexed": self.processed,
                "frames_dropped": self.dropped, "lag_s": None if self.lag_s is None else round(self.lag_s, 1),
                "rows": dict(self.counts)}

    def stop(self) -> None:
        self.stopping.set()
        if self.ffmpeg and self.ffmpeg.poll() is None:
            self.ffmpeg.terminate()

    # ---- the stream -----------------------------------------------------------------------------
    def _run(self) -> None:
        try:
            headers = {"Authorization": f"Bearer {TOKEN.read_text().strip()}"}
            label = self.camera.replace("/", "-")      # Larix needs app/stream, so names can nest (live/phone1)
            with connect(URL.format(camera=label), additional_headers=headers, open_timeout=300,
                         max_size=2 ** 26, ping_interval=20, ping_timeout=60) as ws:
                json.loads(ws.recv(timeout=300))           # "ready": container started, models loaded
                self.state = "indexing"
                receiver = threading.Thread(target=self._receive, args=(ws,), daemon=True)
                receiver.start()
                self._send(ws)
                ws.close()
                receiver.join(timeout=10)
            self.state = "stopped" if self.stopping.is_set() else "stream ended"
        except Exception as e:  # noqa: BLE001 - reported on the session, the API keeps running
            traceback.print_exc()
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"[:300]
        finally:
            if self.ffmpeg and self.ffmpeg.poll() is None:
                self.ffmpeg.kill()
            self.on_change()

    def _send(self, ws) -> None:
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-rtsp_transport", "tcp", "-i", f"rtsp://127.0.0.1:8554/{self.camera}",
             "-an", "-vf", f"fps={FPS},scale=w='min(1920,iw)':h=-2", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "4", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buf = b""
        deadline = self.started + MAX_HOURS * 3600
        while not self.stopping.is_set() and time.time() < deadline:
            chunk = self.ffmpeg.stdout.read1(1 << 16)
            if not chunk:
                break                                  # the stream ended
            buf += chunk
            while (end := buf.find(b"\xff\xd9")) >= 0:  # JPEG end marker; it can't occur inside the image data
                jpeg, buf = buf[:end + 2], buf[end + 2:]
                start = jpeg.find(b"\xff\xd8")
                if start < 0:
                    continue
                if self.sent - self.processed >= IN_FLIGHT:
                    self.dropped += 1                  # the GPU is behind: stay live, skip this frame
                    continue
                ws.send(np.array([time.time()], dtype="<f8").tobytes() + jpeg[start:])
                self.sent += 1

    def _receive(self, ws) -> None:
        last_refresh = time.time()
        with psycopg.connect(DSN) as conn:
            register_vector(conn)
            for msg in ws:
                data = json.loads(msg)
                if data.get("type") != "rows":
                    continue
                self._store(conn, data)
                self.processed += data["received"]
                if data.get("latest_ts"):
                    self.lag_s = time.time() - data["latest_ts"]
                if time.time() - last_refresh > 30:        # let the planner see the live camera
                    self.on_change()
                    last_refresh = time.time()

    # ---- rows -> Postgres -----------------------------------------------------------------------
    def _store(self, conn: psycopg.Connection, d: dict) -> None:
        cam, run = self.camera, self.run
        with conn.cursor() as cur:
            for t in d["tracks"]:
                cur.execute(
                    """insert into tracks (run, camera, track_id, class, conf_mean, start_ts, end_ts, traj_t0,
                                           traj_dt_ms, traj_xywh, best_jpeg)
                       values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       on conflict (run, camera, track_id) do update set class = excluded.class,
                           conf_mean = excluded.conf_mean, end_ts = excluded.end_ts, traj_dt_ms = excluded.traj_dt_ms,
                           traj_xywh = excluded.traj_xywh, best_jpeg = coalesce(excluded.best_jpeg, tracks.best_jpeg)
                       returning id""",
                    (run, cam, t["track_id"], t["class"], t["conf_mean"], _ts(t["start"]), _ts(t["end"]),
                     _ts(t["start"]), t["traj_dt_ms"], t["traj_xywh"], _bytes(t["best_jpeg"])))
                self.track_ids[t["track_id"]] = cur.fetchone()[0]
                self.counts["tracks_updated"] += 1
            for f in d["frames"]:
                cur.execute("""insert into frame_embeds (run, camera, ts, tile, box, motion, thumb, embedding)
                               values (%s, %s, %s, %s, %s, %s, %s, %s)""",
                            (run, cam, _ts(f["ts"]), f["tile"], f["box"], f["motion"], _bytes(f["thumb"]),
                             _vec(f["embedding"])))
                self.counts["scene_tiles"] += 1
            now = time.time()
            waiting, self.waiting = self.waiting, []
            for kind, row in [(k, r) for _, k, r in waiting] + [("crop", r) for r in d["crops"]] + \
                             [("part", r) for r in d["parts"]]:
                track = self.track_ids.get(row["track_id"])
                if track is None:
                    if now - row["ts"] < ORPHAN_S:
                        self.waiting.append((now, kind, row))
                    continue
                if kind == "crop":
                    cur.execute("""insert into track_crops (track, run, camera, ts, bbox, jpeg, embedding)
                                   values (%s, %s, %s, %s, %s, %s, %s)""",
                                (track, run, cam, _ts(row["ts"]), row["bbox"], _bytes(row["jpeg"]), _vec(row["embedding"])))
                    self.counts["crops"] += 1
                else:
                    cur.execute("""insert into part_crops (track, run, camera, part, side, ts, box, px, quality, jpeg,
                                                          embedding)
                                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                                (track, run, cam, row["part"], row["side"], _ts(row["ts"]), row["box"], row["px"],
                                 row["quality"], _bytes(row["jpeg"]), _vec(row["embedding"])))
                    self.counts["close_ups"] += 1
        conn.commit()
