"""Body-part detail index: the details a whole-person crop can't show.

A hand is about 3% of a person crop, so a tattoo, a watch, a phone in the hand or a logo on a shirt
vanish in the person-level embedding. For every tracked person tall enough to show detail:

    native 1080p frame -> person crop (box from the track) -> YOLO26-pose, top-down
      -> part boxes from keypoints: left and right hand (elbow to fingertips), head, torso, legs, feet
      -> per (track, part), the best view in each WINDOW_S window, cut at native resolution
      -> SigLIP 2 embedding; a view nearly identical to the last kept one is dropped
      -> parts.parquet

Sides are the person's own (COCO keypoint convention), not the image's: a person facing the camera
has their right hand on the image's left.

Runs as a second pass over a camera's footage and reuses the first pass's tracks, so track ids stay
stable. PartIndexer.process() only needs frames and person boxes, so it can also run inline.
"""
from __future__ import annotations

import collections
import io
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from geometry import part_boxes
from pipeline import DET_BATCH, Embedder, decode, download

POSE_IMGSZ = 512
POSE_CONF = 0.25
MIN_PERSON_PX = 160      # shorter people can't show part detail at 1080p
PERSON_PAD = 0.12
MATCH_IOU = 0.5          # pose detection <-> track box, within the person crop
WINDOW_S = 1.0           # one view per part per second
MAX_PER_PART = 60        # views kept per (track, part)
DEDUP_COS = 0.96         # a view this similar to the last kept one adds nothing
STALE_S = 2.0            # a held view whose window ended this long ago is emitted
MIN_SIDE = {"hand": 52, "head": 40, "torso": 56, "legs": 40, "feet": 40}   # native pixels; below this a part shows no detail
JPEG_MAX = 320
EMB_BATCH = 64

def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low for motion blur and defocus."""
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return float(lap.var())


def _iou(boxes: np.ndarray, b: tuple[float, float, float, float]) -> np.ndarray:
    ix = np.clip(np.minimum(boxes[:, 2], b[2]) - np.maximum(boxes[:, 0], b[0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, 3], b[3]) - np.maximum(boxes[:, 1], b[1]), 0, None)
    inter = ix * iy
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / np.maximum(area - inter, 1e-6)


def _jpeg(img: Image.Image) -> bytes:
    img = img.copy()
    img.thumbnail((JPEG_MAX, JPEG_MAX))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


class Persons:
    """Tracked person boxes by frame time, from a camera's tracks.parquet."""

    def __init__(self, tracks_parquet: Path):
        self.at: dict[int, list[tuple[int, int, tuple]]] = collections.defaultdict(list)
        for t in pq.read_table(tracks_parquet).to_pylist():
            if t["class"] != "person":
                continue
            xywh = t["traj_xywh"]
            for i, dt in enumerate(t["traj_dt_ms"]):
                ms = t["start_ms"] + dt
                self.at[round(ms / 100)].append((ms, t["track_id"], tuple(xywh[4 * i:4 * i + 4])))

    def boxes(self, ts: float, w: int, h: int) -> list[tuple[int, tuple[float, float, float, float]]]:
        ms = ts * 1000
        out = []
        for pms, tid, (cx, cy, bw, bh) in self.at.get(round(ms / 100), []):
            if abs(pms - ms) <= 60:
                out.append((tid, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
        return out


class PartIndexer:
    def __init__(self, camera: str, pose, embedder: Embedder, persons: Persons):
        self.camera = camera
        self.pose = pose
        self.embedder = embedder
        self.persons = persons
        self.held: dict[tuple, tuple] = {}        # (track, part, side) -> (window, quality, row, image)
        self.pending: list[tuple[tuple, dict, Image.Image]] = []
        self.last_emb: dict[tuple, np.ndarray] = {}
        self.kept: collections.Counter = collections.Counter()
        self.rows: list[dict] = []
        self.timing = collections.Counter()
        self.counts = collections.Counter()

    def process(self, batch: list[tuple[float, np.ndarray]]) -> None:
        """batch: (wall-clock ts, BGR frame) at the tracking rate."""
        crops, meta = [], []
        for ts, bgr in batch:
            h, w = bgr.shape[:2]
            for tid, (x1, y1, x2, y2) in self.persons.boxes(ts, w, h):
                if y2 - y1 < MIN_PERSON_PX:
                    continue
                px, py = (x2 - x1) * PERSON_PAD, (y2 - y1) * PERSON_PAD
                ox, oy = int(max(0, x1 - px)), int(max(0, y1 - py))
                crops.append(bgr[oy:int(min(h, y2 + py)), ox:int(min(w, x2 + px))])
                meta.append((ts, bgr, tid, (x1, y1, x2, y2), ox, oy))
        if crops:
            t = time.perf_counter()
            results = self.pose.predict(crops, imgsz=POSE_IMGSZ, conf=POSE_CONF, half=True, verbose=False)
            self.timing["pose_s"] += time.perf_counter() - t
            self.counts["persons_posed"] += len(crops)
            t = time.perf_counter()
            for (ts, bgr, tid, pbox, ox, oy), res in zip(meta, results):
                self._parts(ts, bgr, tid, pbox, ox, oy, res)
            self.timing["parts_s"] += time.perf_counter() - t
        if batch:
            self._emit_stale(batch[-1][0])
        if len(self.pending) >= EMB_BATCH:
            self._flush()

    def _parts(self, ts, bgr, tid, pbox, ox, oy, res) -> None:
        if res.keypoints is None or not len(res.boxes):
            return
        boxes = res.boxes.xyxy.cpu().numpy() + [ox, oy, ox, oy]
        ious = _iou(boxes, pbox)
        j = int(ious.argmax())
        if ious[j] < MATCH_IOU:
            self.counts["pose_unmatched"] += 1
            return
        self.counts["pose_matched"] += 1
        kp = res.keypoints.data[j].cpu().numpy().astype(np.float64)
        kp[:, 0] += ox
        kp[:, 1] += oy
        h, w = bgr.shape[:2]
        for part, side, (x1, y1, x2, y2), conf in part_boxes(kp, pbox):
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            px = min(x2 - x1, y2 - y1)
            if px < MIN_SIDE[part]:
                self.counts[f"{part}_too_small"] += 1
                continue
            crop = bgr[y1:y2, x1:x2]
            sharp = _sharpness(crop.mean(axis=2, dtype=np.float32))
            quality = min(px, 160) / 160 * conf * sharp / (sharp + 100)
            row = {"track_id": tid, "part": part, "side": side, "ts": ts, "px": px, "conf": conf,
                   "sharpness": sharp, "quality": quality, "box": [x1 / w, y1 / h, x2 / w, y2 / h]}
            self._offer((tid, part, side), ts, quality, row, crop)

    def _offer(self, key: tuple, ts: float, quality: float, row: dict, crop: np.ndarray) -> None:
        self.counts["candidates"] += 1
        win = int(ts // WINDOW_S)
        cur = self.held.get(key)
        if cur and cur[0] != win:
            self._emit(key, cur)
            cur = None
        if cur is None or quality > cur[1]:
            self.held[key] = (win, quality, row, Image.fromarray(np.ascontiguousarray(crop[:, :, ::-1])))

    def _emit_stale(self, now: float) -> None:
        for key in [k for k, v in self.held.items() if now - v[2]["ts"] > STALE_S + WINDOW_S]:
            self._emit(key, self.held[key])

    def _emit(self, key: tuple, held: tuple) -> None:
        del self.held[key]
        if self.kept[key] >= MAX_PER_PART:
            self.counts["capped"] += 1
            return
        self.pending.append((key, held[2], held[3]))

    def _flush(self) -> None:
        if not self.pending:
            return
        t = time.perf_counter()
        embs = self.embedder.images([img for _, _, img in self.pending])
        self.timing["embed_s"] += time.perf_counter() - t
        for (key, row, img), emb in zip(self.pending, embs):
            last = self.last_emb.get(key)
            if last is not None and float(last @ emb) > DEDUP_COS:
                self.counts["deduped"] += 1
                continue
            if self.kept[key] >= MAX_PER_PART:
                self.counts["capped"] += 1
                continue
            self.last_emb[key] = emb
            self.kept[key] += 1
            row["jpeg"] = _jpeg(img)
            row["embedding"] = emb.astype(np.float32)
            self.rows.append(row)
        self.pending.clear()

    def finish(self) -> None:
        for key in list(self.held):
            self._emit(key, self.held[key])
        self._flush()

    # ---- driver: a second pass over the footage -------------------------------------------------
    def run(self, clips: list[dict], workdir: Path) -> None:
        t = time.perf_counter()
        with ThreadPoolExecutor(6) as ex:
            paths = list(ex.map(lambda c: download(c, workdir), clips))
        self.timing["download_s"] = time.perf_counter() - t

        frames: queue.Queue = queue.Queue(maxsize=DET_BATCH * 4)
        threading.Thread(target=decode, args=(clips, paths, frames, self.counts), daemon=True).start()
        batch = []
        while True:
            t = time.perf_counter()
            item = frames.get()
            self.timing["wait_decode_s"] += time.perf_counter() - t
            if item == "gap":
                continue    # parts need no tracker state
            if item is None:
                break
            batch.append((item[0], item[1]))
            if len(batch) == DET_BATCH:
                self.process(batch)
                batch = []
        self.process(batch)
        self.finish()

    def write(self, out_dir: Path, video_seconds: float, total_s: float) -> dict:
        rows = self.rows
        pq.write_table(pa.table({
            "camera": [self.camera] * len(rows),
            "track_id": [r["track_id"] for r in rows],
            "part": [r["part"] for r in rows],
            "side": [r["side"] for r in rows],
            "ts_ms": [int(r["ts"] * 1000) for r in rows],
            "box": [r["box"] for r in rows],
            "px": [r["px"] for r in rows],
            "conf": [r["conf"] for r in rows],
            "quality": [r["quality"] for r in rows],
            "jpeg": [r["jpeg"] for r in rows],
            "embedding": pa.array([r["embedding"] for r in rows], type=pa.list_(pa.float32())),
        }), out_dir / "parts.parquet")
        by_part = collections.Counter(r["part"] + (r["side"] or "") for r in rows)
        stats = {
            "camera": self.camera,
            "video_s": round(video_seconds),
            "total_s": round(total_s, 1),
            "realtime_x": round(video_seconds / total_s, 1),
            "timing_s": {k: round(v, 1) for k, v in self.timing.items()},
            "counts": dict(self.counts) | {"parts": len(rows), "by_part": dict(by_part),
                                           "tracks_with_parts": len({r["track_id"] for r in rows})},
            "gpu": torch.cuda.get_device_name(0),
        }
        (out_dir / "parts_stats.json").write_text(json.dumps(stats, indent=2))
        return stats
