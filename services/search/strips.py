"""Dense frame strips from the 1080p originals, for verifying actions.

An action is change over a few seconds, so one frame can't show it. A strip is STRIP_FRAMES frames
spanning +/- STRIP_HALF_S around the candidate's moment, cropped to the region the candidate moves
through (union of its tracked boxes over the window) expanded for context, with the candidate
outlined in every frame where it was tracked.
"""
import csv
import io
import json
import subprocess
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image, ImageDraw

RAW = Path.home() / "data/meva/raw"
UPLOADS = Path.home() / "data/uploads"
LIVE = Path("/var/lib/godseye/live")         # MediaMTX recordings of live streams
PLAYBACK = "http://127.0.0.1:9996/get?{query}"
LIVE_WINDOW_S = 30.0
STRIP_HALF_S = 2.0
STRIP_FPS = 2            # 8 frames over 4 s
CONTEXT = 1.6            # region side = this x the candidate's union box side
MIN_REGION = 0.12        # ... but at least this fraction of the frame per side (small people stay large)
OUT_PX = 448             # long side of each strip frame
BOX_MATCH_S = 0.3        # a tracked box within this of a frame's time is drawn on it


@dataclass
class Frame:
    offset_s: float      # relative to the candidate's moment
    jpeg: bytes


class Footage:
    """Maps (camera, wall-clock time) to a clip file and offset: the MEVA downloader's manifest, plus
    uploaded videos that finished indexing (one clip each)."""

    def __init__(self, raw: Path = RAW, uploads: Path = UPLOADS):
        self.raw, self.uploads = raw, uploads
        self.refresh()

    def refresh(self) -> None:
        clips: dict[str, list[tuple[datetime, datetime, Path]]] = {}
        if (self.raw / "manifest.csv").exists():
            for r in csv.DictReader(open(self.raw / "manifest.csv")):
                start = datetime.fromisoformat(r["start"]).replace(tzinfo=timezone.utc)
                end = datetime.fromisoformat(r["end"]).replace(tzinfo=timezone.utc)
                clips.setdefault(r["camera"], []).append((start, end, self.raw / r["file"]))
        for f in self.uploads.glob("*/source.json"):
            src = json.loads(f.read_text())
            if src.get("status") == "ready":
                clips[src["camera"]] = [(datetime.fromisoformat(src["start"]), datetime.fromisoformat(src["end"]),
                                         f.parent / "original.mp4")]
        for v in clips.values():
            v.sort()
        self.clips = clips

    def locate(self, camera: str, ts: datetime) -> tuple[Path | str, float, float] | None:
        """(clip path, offset of ts in seconds, clip duration in seconds). For a live stream the clip is a
        URL on MediaMTX's playback server: a LIVE_WINDOW_S window of its recording centred on ts."""
        if camera not in self.clips and (LIVE / camera).is_dir():
            start = ts - timedelta(seconds=LIVE_WINDOW_S / 2)
            url = PLAYBACK.format(query=urlencode({"path": camera, "start": start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                                   "duration": LIVE_WINDOW_S}))
            return url, LIVE_WINDOW_S / 2, LIVE_WINDOW_S
        clips = self.clips.get(camera, [])
        i = bisect_right([c[0] for c in clips], ts) - 1
        if i < 0 or ts >= clips[i][1]:
            return None
        start, end, path = clips[i]
        return path, (ts - start).total_seconds(), (end - start).total_seconds()


def region(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    """Normalized x, y, w, h of the zoomed region around the union of boxes (x1, y1, x2, y2)."""
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    w = min(1.0, max((x2 - x1) * CONTEXT, MIN_REGION))
    h = min(1.0, max((y2 - y1) * CONTEXT, MIN_REGION))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x = min(max(cx - w / 2, 0.0), 1.0 - w)
    y = min(max(cy - h / 2, 0.0), 1.0 - h)
    return x, y, w, h


def _jpegs(stream: bytes) -> list[bytes]:
    # MJPEG frames from image2pipe: each starts with SOI (FFD8); entropy-coded data never contains it.
    parts = stream.split(b"\xff\xd8")
    return [b"\xff\xd8" + p for p in parts if p]


def extract(footage: Footage, camera: str, moment: datetime, area: tuple[float, float, float, float],
            track_boxes: list[tuple[datetime, tuple[float, float, float, float]]] | None = None,
            half_s: float = STRIP_HALF_S, fps: float = STRIP_FPS, out_px: int = OUT_PX) -> list[Frame]:
    """Frames around `moment` cropped to `area` (normalized x, y, w, h), candidate outlined if tracked."""
    loc = footage.locate(camera, moment)
    if loc is None:
        return []
    path, offset, duration = loc
    start = max(0.0, offset - half_s)
    end = min(duration, offset + half_s)
    x, y, w, h = area
    scale = f"scale='if(gt(iw,ih),{out_px},-2)':'if(gt(iw,ih),-2,{out_px})'"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-fflags", "+genpts", "-ss", f"{start:.3f}", "-i", str(path),
           "-t", f"{end - start:.3f}", "-vf", f"fps={fps},crop=iw*{w:.4f}:ih*{h:.4f}:iw*{x:.4f}:ih*{y:.4f},{scale}",
           "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "3", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=30).stdout
    frames = []
    for i, jpeg in enumerate(_jpegs(out)):
        t = start + i / fps
        ts = moment + timedelta(seconds=t - offset)
        if track_boxes:
            near = min(track_boxes, key=lambda p: abs((p[0] - ts).total_seconds()))
            if abs((near[0] - ts).total_seconds()) <= BOX_MATCH_S:
                jpeg = _outline(jpeg, near[1], area)
        frames.append(Frame(round(t - offset, 1), jpeg))
    return frames


def _outline(jpeg: bytes, box: tuple[float, float, float, float], area: tuple[float, float, float, float]) -> bytes:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    W, H = img.size
    x, y, w, h = area
    rect = [(box[0] - x) / w * W, (box[1] - y) / h * H, (box[2] - x) / w * W, (box[3] - y) / h * H]
    ImageDraw.Draw(img).rectangle(rect, outline=(255, 0, 0), width=2)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()
