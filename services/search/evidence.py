"""Evidence images for the verifier.

Formats, chosen by the planner's intent:
  appearance  stored crops of the candidate + one zoomed 1080p frame at the moment with the candidate
              outlined.
  state       4 zoomed frames 1 s apart (an ongoing activity shows in any frame).
  action      8 zoomed frames 0.5 s apart around the moment (a brief change needs dense frames).
Strips (strips.py) outline the candidate in each frame. Whatever the intent, each body part a
condition is about (hand, head, torso, feet) adds the candidate's best close-ups of it: native-
resolution crops, ranked against that condition (the asked-for side first), upscaled so the
verifier gets enough pixels. A whole-scene frame comes last so the verifier sees where it happens.
Captions carry each image's time offset, so order is explicit. Without local footage (clip not
downloaded) everything falls back to stored crops and thumbnails.
"""
import io
from datetime import datetime, timedelta

import numpy as np
import psycopg
from pgvector import HalfVector
from PIL import Image, ImageDraw

from search import strips
from search.retrieve import Event, Query

APPEARANCE_CROPS = 3
PART_VIEWS = 4           # close-ups per part a condition is about
DETAIL_APART_S = 0.9     # views of one person at least this far apart in time
DETAIL_MIN_PX = 384      # small crops are upscaled to this so the verifier gets enough image tokens
SIDES = {"L": "LEFT", "R": "RIGHT"}
CONTEXT_WITHIN = timedelta(seconds=2)
_footage: strips.Footage | None = None


def footage() -> strips.Footage:
    global _footage
    if _footage is None:
        _footage = strips.Footage()
    return _footage


def _outlined(jpeg: bytes, box: list[float]) -> bytes:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = img.size
    ImageDraw.Draw(img).rectangle([box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline=(255, 0, 0), width=3)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _spread(rows: list, n: int) -> list:
    if len(rows) <= n:
        return rows
    step = (len(rows) - 1) / (n - 1)
    return [rows[round(i * step)] for i in range(n)]


def track_points(conn: psycopg.Connection, track: int) -> list[tuple[datetime, tuple[float, float, float, float]]]:
    """(time, normalized x1, y1, x2, y2) for every tracked point of a track."""
    t0, dts, xywh = conn.execute("select traj_t0, traj_dt_ms, traj_xywh from tracks where id = %s",
                                 (track,)).fetchone()
    pts = []
    for i, dt in enumerate(dts):
        cx, cy, w, h = xywh[4 * i:4 * i + 4]
        pts.append((t0 + timedelta(milliseconds=dt), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return pts


def _scene(conn: psycopg.Connection, ev: Event) -> tuple[str, bytes] | None:
    row = conn.execute(
        """select ts, thumb from frame_embeds where camera = %s and tile = 0 and thumb is not null
           and ts between %s and %s order by abs(extract(epoch from ts - %s)) limit 1""",
        (ev.camera, ev.moment - CONTEXT_WITHIN, ev.moment + CONTEXT_WITHIN, ev.moment)).fetchone()
    if not row:
        return None
    return (f"Whole scene at t{(row[0] - ev.moment).total_seconds():+.1f}s (candidate area in red):",
            _outlined(row[1], ev.bbox))


def gather(conn: psycopg.Connection, ev: Event, intent: str, parts: list[Query] = ()) -> list[tuple[str, bytes]]:
    """Evidence by intent, plus close-ups of each body part a condition is about (`parts`)."""
    pts = track_points(conn, ev.track) if ev.kind == "track" else []
    window = [b for t, b in pts if abs((t - ev.moment).total_seconds()) <= strips.STRIP_HALF_S]
    area = strips.region(window or [tuple(ev.bbox)])
    out: list[tuple[str, bytes]] = []

    if intent in ("state", "action"):
        fps = 1 if intent == "state" else strips.STRIP_FPS
        frames = strips.extract(footage(), ev.camera, ev.moment, area, pts or None, fps=fps)
        out += [(f"Frame at t{f.offset_s:+.1f}s (zoomed{', candidate in red' if pts else ''}):", f.jpeg)
                for f in frames]
    else:
        if ev.kind == "track":
            crops = conn.execute("select ts, jpeg from track_crops where track = %s order by ts", (ev.track,)).fetchall()
            out += [(f"Close-up of the candidate at t{(ts - ev.moment).total_seconds():+.0f}s:", jpeg)
                    for ts, jpeg in _spread(crops, APPEARANCE_CROPS - (1 if parts else 0))]
        frames = strips.extract(footage(), ev.camera, ev.moment, area, pts or None, half_s=0.25)
        out += [(f"Zoomed frame at t{f.offset_s:+.1f}s{' (candidate in red)' if pts else ''}:", f.jpeg)
                for f in frames[:1]]

    if not out:  # no local footage for this moment: stored images only
        crops = conn.execute("select ts, jpeg from track_crops where track = %s order by ts",
                             (ev.track,)).fetchall() if ev.kind == "track" else []
        out += [(f"Close-up at t{(ts - ev.moment).total_seconds():+.0f}s:", jpeg) for ts, jpeg in _spread(crops, 6)]

    if ev.kind == "track":
        for q in {q.scale: q for q in parts}.values():       # one set of close-ups per part
            side = q.side if q.scale == "hand" else None
            for _, ts, vside, px, _, jpeg in detail_views(conn, ev.track, q.scale, q.vec, side, n=PART_VIEWS):
                what = f"the person's {SIDES[vside]} hand (pose estimate)" if vside else f"the {q.scale}"
                out.append((f"Close-up of {what} for '{q.phrase}', at t{(ts - ev.moment).total_seconds():+.1f}s, "
                            f"{px} px wide in the original:", _upscaled(bytes(jpeg))))
    scene = _scene(conn, ev)
    if scene:
        out.append(scene)
    return out


def _upscaled(jpeg: bytes, min_px: int = DETAIL_MIN_PX) -> bytes:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if max(img.size) >= min_px:
        return jpeg
    k = min_px / max(img.size)
    img = img.resize((round(img.size[0] * k), round(img.size[1] * k)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def detail_views(conn: psycopg.Connection, track: int, part: str, qvec: np.ndarray, side: str | None = None,
                 n: int = PART_VIEWS) -> list[tuple]:
    """A track's best views of a part for one condition (id, ts, side, px, box, jpeg), spread in time.

    Every stored view of the part is ranked against the condition; when it names a side, that side's
    views come first, with room left for one view of the other side."""
    rows = conn.execute("""select id, ts, side, px, box, jpeg from part_crops where track = %s and part = %s
                           order by embedding <#> %s limit 80""",
                        (track, part, HalfVector(qvec.astype(np.float16)))).fetchall()
    picked: list[tuple] = []

    def take(pool: list[tuple], upto: int) -> None:
        for r in pool:
            if len(picked) >= upto:
                return
            if r not in picked and all(abs((r[1] - q[1]).total_seconds()) >= DETAIL_APART_S for q in picked):
                picked.append(r)

    take([r for r in rows if side and r[2] == side], n - 1 if side else n)
    take(rows, n)
    return picked
