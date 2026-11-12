"""Live indexing: the batch indexer's two passes, run second by second on a stream.

The Studio sends JPEG frames with their wall-clock time (~5 fps) over a websocket. Each second's frames
go through detection and tracking, crops and scene tiles (pipeline.CameraIndexer), and body-part views
(parts.PartIndexer, fed from the live tracker instead of a finished tracks.parquet). Everything new goes
back as rows for the Studio to insert into Postgres. Tracks keep growing, so a track that changed is sent
again whole (its trajectory so far) and the Studio upserts it.
"""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from parts import PartIndexer
from pipeline import CameraIndexer, Embedder

FORGET_S = 60.0      # a track not seen for this long is final: sent for the last time and dropped from memory


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _emb(v: np.ndarray) -> str:
    return _b64(np.asarray(v, dtype=np.float16).tobytes())


class LivePersons:
    """Person boxes at a frame time, from the live tracker (the batch pass reads them from Parquet)."""

    def __init__(self, indexer: CameraIndexer):
        self.ix = indexer

    def boxes(self, ts: float, w: int, h: int) -> list[tuple[int, tuple[float, float, float, float]]]:
        out = []
        for tid, t in self.ix.tracks.items():
            if t.classes.most_common(1)[0][0] != "person":
                continue
            for i in range(len(t.ts) - 1, max(-1, len(t.ts) - 12), -1):   # the batch's frames are the last few
                if abs(t.ts[i] - ts) < 0.06:
                    cx, cy, bw, bh = t.xywh[i]
                    out.append((tid, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
                    break
        return out


class LiveIndexer:
    def __init__(self, camera: str, detector, pose, embedder: Embedder):
        self.cam = CameraIndexer(camera, detector, embedder)
        self.parts = PartIndexer(camera, pose, embedder, LivePersons(self.cam))
        self.sent_points: dict[int, int] = {}       # track id -> trajectory points already sent
        self.sent_best: dict[int, float] = {}       # track id -> best-crop score already sent
        self.frames = 0

    def process(self, frames: list[tuple[float, bytes]]) -> dict:
        """frames: (wall-clock ts, JPEG) in time order. Returns the new and changed rows."""
        t0 = time.perf_counter()
        batch = []
        for ts, jpeg in frames:
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
            batch.append((ts, bgr, gray))
        if batch:
            self.cam._detect_and_track(batch)
            self.cam._flush()
            self.parts.process([(ts, bgr) for ts, bgr, _ in batch])
            self.parts._flush()
            self.frames += len(batch)
        rows = self._rows(batch[-1][0] if batch else time.time())
        rows["ms"] = round((time.perf_counter() - t0) * 1000)
        rows["latest_ts"] = batch[-1][0] if batch else None
        return rows

    def _rows(self, now: float) -> dict:
        tracks = []
        for tid, t in list(self.cam.tracks.items()):
            if len(t.ts) >= 2 and len(t.ts) != self.sent_points.get(tid):
                best = t.best_jpeg if t.best_score > self.sent_best.get(tid, -1.0) else None
                tracks.append({"track_id": tid, "class": t.classes.most_common(1)[0][0],
                               "conf_mean": float(np.mean(t.confs)), "start": t.ts[0], "end": t.ts[-1],
                               "traj_dt_ms": [int((x - t.ts[0]) * 1000) for x in t.ts],
                               "traj_xywh": [round(float(v), 5) for box in t.xywh for v in box],
                               "best_jpeg": _b64(best) if best else None})
                self.sent_points[tid] = len(t.ts)
                if best:
                    self.sent_best[tid] = t.best_score
            if t.ts and now - t.ts[-1] > FORGET_S:
                del self.cam.tracks[tid]
        crops = [{"track_id": r["track_id"], "ts": r["ts"], "bbox": [float(v) for v in r["bbox"]],
                  "jpeg": _b64(r["jpeg"]), "embedding": _emb(r["embedding"])} for r in self.cam.crop_rows]
        frames = [{"ts": r["ts"], "tile": r["tile"], "box": [float(v) for v in r["box"]], "motion": r["motion"],
                   "thumb": _b64(r["thumb"]) if r["thumb"] else None, "embedding": _emb(r["embedding"])}
                  for r in self.cam.frame_rows]
        parts = [{"track_id": r["track_id"], "part": r["part"], "side": r["side"], "ts": r["ts"],
                  "box": [float(v) for v in r["box"]], "px": int(r["px"]), "quality": float(r["quality"]),
                  "jpeg": _b64(r["jpeg"]), "embedding": _emb(r["embedding"])} for r in self.parts.rows]
        self.cam.crop_rows.clear()
        self.cam.frame_rows.clear()
        self.parts.rows.clear()
        return {"tracks": tracks, "crops": crops, "frames": frames, "parts": parts}
