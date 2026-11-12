"""Per-camera indexing, run inside a Modal GPU container.

    decode (every frame, CPU thread)
      -> loop A, fixed DET_FPS, never motion-gated: YOLO26 -> ByteTrack -> tracks + trajectories
      -> loop B, change-driven: track crops + full frame & 2x2 tiles -> SigLIP 2 embeddings
      -> Parquet (tracks, crops, frames) + stats.json

A camera's clips are processed as one continuous stream: tracker state carries across 5-minute
files, and a recording gap > GAP_S resets it (§2.2).
"""
from __future__ import annotations

import collections
import io
import json
import queue
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import supervision as sv
import torch
from PIL import Image

BUCKET = "https://mevadata-public-01.s3.amazonaws.com"
SRC_FPS = 30
DET_FPS = 5
STRIDE = SRC_FPS // DET_FPS
GAP_S = 2.0

# COCO classes worth tracking in CCTV; anything else is found through tiles and captions.
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
           24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase"}
DET_IMGSZ = 1280      # wide shots: people are small
DET_CONF = 0.2
DET_BATCH = 16

CROP_EVERY_S = 2.0          # while the track moves
STATIC_CROP_EVERY_S = 10.0  # while it stays put (seated people dominate crowded indoor scenes)
STATIC_MOVE = 0.02          # center shift, as a fraction of the frame, that counts as moving
MAX_CROPS_PER_TRACK = 12
MIN_CROP_PX = 16
CROP_PAD = 0.15

FRAME_EVERY_S = 2.0   # at most this often while the scene changes
KEEPALIVE_S = 30.0    # and at least this often when it doesn't
MOTION_THRESHOLD = 2.0
EMB_BATCH = 64


@dataclass
class Track:
    track_id: int
    classes: collections.Counter = field(default_factory=collections.Counter)
    ts: list[float] = field(default_factory=list)      # seconds since epoch
    xywh: list[tuple[float, float, float, float]] = field(default_factory=list)  # normalized
    confs: list[float] = field(default_factory=list)
    last_crop_ts: float = -1e9
    last_crop_xy: tuple[float, float] = (-1.0, -1.0)
    n_crops: int = 0
    best_score: float = -1.0
    best_jpeg: bytes = b""


def _jpeg(img: Image.Image, max_side: int, quality: int = 80) -> bytes:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _square(img: Image.Image) -> Image.Image:
    """Pad to a square so SigLIP's fixed-size resize doesn't squash tall people or wide tiles."""
    w, h = img.size
    side = max(w, h)
    out = Image.new("RGB", (side, side), (128, 128, 128))
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


class Embedder:
    def __init__(self, name: str, cache_dir: str):
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(name, cache_dir=cache_dir, dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def images(self, imgs: list[Image.Image]) -> np.ndarray:
        pixels = self.processor(images=[_square(i) for i in imgs], return_tensors="pt")["pixel_values"]
        out = self.model.get_image_features(pixel_values=pixels.cuda().half())
        out = getattr(out, "pooler_output", out)
        return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()


def download(clip: dict, workdir: Path) -> Path:
    """A clip into the work dir: from the MEVA bucket, or (uploads) from a path on the index volume."""
    dest = workdir / clip["file"]
    if "path" in clip:
        shutil.copyfile(clip["path"], dest)
    else:
        urllib.request.urlretrieve(f"{BUCKET}/{clip['key']}", dest)
    return dest


def decode(clips: list[dict], paths: list[Path], out: queue.Queue, counts: collections.Counter) -> None:
    """Every STRIDE-th frame as (wall-clock ts, BGR, 160x90 gray), "gap" at recording gaps, None at the end.
    Clips are deleted once decoded."""
    prev_end = None
    for clip, path in zip(clips, paths):
        start = datetime.fromisoformat(clip["start"])
        if prev_end is not None and (start - prev_end).total_seconds() > GAP_S:
            out.put("gap")
        n = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                n = i + 1
                counts["frames_decoded"] += 1
                if i % STRIDE:
                    continue
                bgr = frame.to_ndarray(format="bgr24")
                small = frame.reformat(width=160, height=90, format="gray").to_ndarray().astype(np.float32)
                ts = (start + timedelta(seconds=i / SRC_FPS)).timestamp()
                out.put((ts, bgr, small))
                counts["det_frames"] += 1
        prev_end = start + timedelta(seconds=n / SRC_FPS)
        path.unlink(missing_ok=True)
    out.put(None)


class CameraIndexer:
    def __init__(self, camera: str, detector, embedder: Embedder):
        self.camera = camera
        self.detector = detector
        self.embedder = embedder
        self.tracks: dict[int, Track] = {}
        self.crop_rows: list[dict] = []
        self.frame_rows: list[dict] = []
        self.pending: list[tuple[str, dict, Image.Image]] = []  # (kind, row, image) awaiting embedding
        self.last_frame_ts = -1e9
        self.last_frame_gray: np.ndarray | None = None
        self.timing = collections.Counter()
        self.counts = collections.Counter()
        self.track_offset = 0
        self._new_tracker()

    def _new_tracker(self) -> None:
        # Track ids restart after a reset; the offset keeps them unique within the camera.
        self.track_offset = max(self.tracks, default=0) + 1
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=DET_FPS * 4,
                                    minimum_matching_threshold=0.8, frame_rate=DET_FPS,
                                    minimum_consecutive_frames=2)

    # ---- loop A: detection + tracking -------------------------------------------------------
    def _detect_and_track(self, batch: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
        t = time.perf_counter()
        results = self.detector.predict([bgr for _, bgr, _ in batch], imgsz=DET_IMGSZ, conf=DET_CONF,
                                        classes=list(CLASSES), half=True, verbose=False)
        self.timing["detect_s"] += time.perf_counter() - t
        for (ts, bgr, gray), res in zip(batch, results):
            t = time.perf_counter()
            dets = self.tracker.update_with_detections(sv.Detections.from_ultralytics(res))
            self.timing["track_s"] += time.perf_counter() - t
            self.counts["detections"] += len(dets)
            h, w = bgr.shape[:2]
            rgb = None
            for (x1, y1, x2, y2), cls, conf, tid in zip(dets.xyxy, dets.class_id, dets.confidence, dets.tracker_id):
                if tid < 0:  # not yet confirmed (minimum_consecutive_frames)
                    continue
                tid = int(tid) + self.track_offset
                tr = self.tracks.setdefault(tid, Track(tid))
                tr.classes[CLASSES[int(cls)]] += 1
                tr.ts.append(ts)
                tr.xywh.append(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
                tr.confs.append(float(conf))
                if min(x2 - x1, y2 - y1) < MIN_CROP_PX:
                    continue
                score = float(conf) * (x2 - x1) * (y2 - y1)
                cx, cy = tr.xywh[-1][:2]
                moved = max(abs(cx - tr.last_crop_xy[0]), abs(cy - tr.last_crop_xy[1])) > STATIC_MOVE
                since = ts - tr.last_crop_ts
                wants_crop = tr.n_crops < MAX_CROPS_PER_TRACK and (
                    (moved and since >= CROP_EVERY_S) or since >= STATIC_CROP_EVERY_S)
                if not wants_crop and score <= tr.best_score:
                    continue
                if rgb is None:
                    rgb = Image.fromarray(bgr[:, :, ::-1])
                px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
                box = (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))
                crop = rgb.crop(tuple(int(v) for v in box))
                if score > tr.best_score:
                    tr.best_score, tr.best_jpeg = score, _jpeg(crop, 256)
                if wants_crop:
                    tr.last_crop_ts, tr.last_crop_xy, tr.n_crops = ts, (cx, cy), tr.n_crops + 1
                    row = {"track_id": tid, "ts": ts, "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                           "jpeg": _jpeg(crop, 256)}
                    self.pending.append(("crop", row, crop))
            self._maybe_index_frame(ts, bgr, gray, rgb)
        if len(self.pending) >= EMB_BATCH:
            self._flush()

    # ---- loop B: change-driven frames and tiles -------------------------------------------------
    def _maybe_index_frame(self, ts: float, bgr: np.ndarray, gray: np.ndarray, rgb: Image.Image | None) -> None:
        dt = ts - self.last_frame_ts
        motion = 0.0 if self.last_frame_gray is None else float(np.abs(gray - self.last_frame_gray).mean())
        if not ((motion > MOTION_THRESHOLD and dt >= FRAME_EVERY_S) or dt >= KEEPALIVE_S):
            return
        self.last_frame_ts, self.last_frame_gray = ts, gray
        if rgb is None:
            rgb = Image.fromarray(bgr[:, :, ::-1])
        w, h = rgb.size
        tw, th = int(w * 0.6), int(h * 0.6)
        boxes = [(0, 0, w, h), (0, 0, tw, th), (w - tw, 0, w, th), (0, h - th, tw, h), (w - tw, h - th, w, h)]
        for tile, box in enumerate(boxes):
            row = {"ts": ts, "tile": tile, "box": [box[0] / w, box[1] / h, box[2] / w, box[3] / h],
                   "motion": motion, "thumb": _jpeg(rgb, 480, 70) if tile == 0 else None}
            self.pending.append(("frame", row, rgb.crop(box)))

    def _flush(self) -> None:
        if not self.pending:
            return
        t = time.perf_counter()
        embs = self.embedder.images([img for _, _, img in self.pending])
        self.timing["embed_s"] += time.perf_counter() - t
        for (kind, row, _), emb in zip(self.pending, embs):
            row["embedding"] = emb.astype(np.float32)
            (self.crop_rows if kind == "crop" else self.frame_rows).append(row)
        self.pending.clear()

    # ---- driver -----------------------------------------------------------------------------------
    def run(self, clips: list[dict], workdir: Path) -> None:
        t_dl = time.perf_counter()
        with ThreadPoolExecutor(6) as ex:
            paths = list(ex.map(lambda c: download(c, workdir), clips))
        self.timing["download_s"] = time.perf_counter() - t_dl

        frames: queue.Queue = queue.Queue(maxsize=DET_BATCH * 4)
        threading.Thread(target=decode, args=(clips, paths, frames, self.counts), daemon=True).start()
        batch = []
        while True:
            t = time.perf_counter()
            item = frames.get()
            self.timing["wait_decode_s"] += time.perf_counter() - t
            if item == "gap":
                if batch:
                    self._detect_and_track(batch)
                batch = []
                self._new_tracker()
                self.counts["gaps"] += 1
                continue
            if item is None:
                break
            batch.append(item)
            if len(batch) == DET_BATCH:
                self._detect_and_track(batch)
                batch = []
        if batch:
            self._detect_and_track(batch)
        self._flush()

    # ---- output -----------------------------------------------------------------------------------
    def write(self, out_dir: Path, video_seconds: float, total_s: float) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_t = pa.list_(pa.float32())
        ts_ms = lambda s: int(s * 1000)

        tracks = [t for t in self.tracks.values() if len(t.ts) >= 2]
        pq.write_table(pa.table({
            "camera": [self.camera] * len(tracks),
            "track_id": [t.track_id for t in tracks],
            "class": [t.classes.most_common(1)[0][0] for t in tracks],
            "conf_mean": [float(np.mean(t.confs)) for t in tracks],
            "start_ms": [ts_ms(t.ts[0]) for t in tracks],
            "end_ms": [ts_ms(t.ts[-1]) for t in tracks],
            "traj_dt_ms": [[ts_ms(x - t.ts[0]) for x in t.ts] for t in tracks],
            "traj_xywh": [[v for box in t.xywh for v in box] for t in tracks],
            "best_jpeg": [t.best_jpeg for t in tracks],
        }), out_dir / "tracks.parquet")

        keep = {t.track_id for t in tracks}
        crops = [r for r in self.crop_rows if r["track_id"] in keep]
        pq.write_table(pa.table({
            "camera": [self.camera] * len(crops),
            "track_id": [r["track_id"] for r in crops],
            "ts_ms": [ts_ms(r["ts"]) for r in crops],
            "bbox": [r["bbox"] for r in crops],
            "jpeg": [r["jpeg"] for r in crops],
            "embedding": pa.array([r["embedding"] for r in crops], type=emb_t),
        }), out_dir / "crops.parquet")

        fr = self.frame_rows
        pq.write_table(pa.table({
            "camera": [self.camera] * len(fr),
            "ts_ms": [ts_ms(r["ts"]) for r in fr],
            "tile": [r["tile"] for r in fr],
            "box": [r["box"] for r in fr],
            "motion": [r["motion"] for r in fr],
            "thumb": [r["thumb"] for r in fr],
            "embedding": pa.array([r["embedding"] for r in fr], type=emb_t),
        }), out_dir / "frames.parquet")

        stats = {
            "camera": self.camera,
            "video_s": round(video_seconds),
            "total_s": round(total_s, 1),
            "realtime_x": round(video_seconds / total_s, 1),
            "timing_s": {k: round(v, 1) for k, v in self.timing.items()},
            "counts": dict(self.counts) | {"tracks": len(tracks), "crops": len(crops),
                                           "frame_embeds": len(fr),
                                           "track_classes": dict(collections.Counter(
                                               t.classes.most_common(1)[0][0] for t in tracks))},
            "gpu": torch.cuda.get_device_name(0),
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        return stats
