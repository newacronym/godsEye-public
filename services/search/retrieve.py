"""Provisional retrieval.

kNN over track crops and frame tiles, then grouping into events. Every event has an uncapped
event interval (track span, or merged contiguous frame hits) and a representative moment (the
best-scoring timestamp). The two ranked lists are fused with reciprocal rank fusion.

Appearance questions are split into conditions, each at its own scale (search_attributes): every
tracked person or object is scored exactly on each condition, against whole-object crops or the
body-part views (hands, head, torso, feet), and the scores are combined.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import psycopg
from pgvector import HalfVector

RRF_K = 60
WINDOW_MERGE_S = 3.0   # frame hits closer than this on one camera form one window
WINDOW_PAD_S = 1.0


@dataclass
class Event:
    camera: str
    kind: str                      # "track" | "window"
    start: datetime                # event interval, uncapped
    end: datetime
    moment: datetime               # representative moment
    score: float                   # best cosine similarity
    bbox: list[float] | None       # normalized x1, y1, x2, y2 at the moment (crop box or tile box)
    track: int | None = None       # tracks.id for track events
    cls: str | None = None
    evidence: list[int] = field(default_factory=list)   # track_crops.id, frame_embeds.id or part_crops.id
    rrf: float = 0.0
    part: str | None = None        # detail events: the body part searched
    part_box: list[float] | None = None    # ... its box at the moment
    part_side: str | None = None   # ... L or R (the person's own) for hands
    condition_scores: list[float | None] = field(default_factory=list)   # attribute search: similarity per condition
    sightings: list[int] = field(default_factory=list)   # tracks.id of fragments merged into this result


def _filters(cameras: list[str] | None, t0: datetime | None, t1: datetime | None, alias: str) -> tuple[str, list]:
    sql, args = [], []
    if cameras:
        sql.append(f"{alias}.camera = any(%s)")
        args.append(cameras)
    if t0:
        sql.append(f"{alias}.ts >= %s")
        args.append(t0)
    if t1:
        sql.append(f"{alias}.ts < %s")
        args.append(t1)
    return (" and ".join(sql) and "where " + " and ".join(sql)), args


def search(conn: psycopg.Connection, qvec: np.ndarray, k: int = 300, limit: int = 30,
           cameras: list[str] | None = None, t0: datetime | None = None, t1: datetime | None = None) -> list[Event]:
    q = HalfVector(qvec.astype(np.float16))
    conn.execute("set hnsw.ef_search = 400")
    conn.execute("set hnsw.iterative_scan = relaxed_order")  # keep k results when filters are selective

    where, args = _filters(cameras, t0, t1, "c")
    crops = conn.execute(
        f"""select c.id, c.track, c.camera, c.ts, c.bbox, -(c.embedding <#> %s) as score,
                   t.start_ts, t.end_ts, t.class
            from (select * from track_crops c {where} order by c.embedding <#> %s limit %s) c
            join tracks t on t.id = c.track""", [q, *args, q, k]).fetchall()

    where, args = _filters(cameras, t0, t1, "f")
    frames = conn.execute(
        f"""select f.id, f.camera, f.ts, f.box, -(f.embedding <#> %s) as score
            from frame_embeds f {where} order by f.embedding <#> %s limit %s""", [q, *args, q, k]).fetchall()

    track_events: dict[int, Event] = {}
    for cid, track, cam, ts, bbox, score, start, end, cls in crops:
        ev = track_events.get(track)
        if ev is None:
            track_events[track] = Event(cam, "track", start, end, ts, score, bbox, track, cls, [cid])
        else:
            ev.evidence.append(cid)
            if score > ev.score:
                ev.score, ev.moment, ev.bbox = score, ts, bbox

    windows: list[Event] = []
    for fid, cam, ts, box, score in sorted(frames, key=lambda r: (r[1], r[2])):
        last = windows[-1] if windows else None
        if last and last.camera == cam and (ts - last.end).total_seconds() <= WINDOW_MERGE_S:
            last.end = max(last.end, ts)
            last.evidence.append(fid)
            if score > last.score:
                last.score, last.moment, last.bbox = score, ts, box
        else:
            windows.append(Event(cam, "window", ts, ts, ts, score, box, evidence=[fid]))
    pad = timedelta(seconds=WINDOW_PAD_S)
    for w in windows:
        w.start, w.end = w.start - pad, w.end + pad

    ranked = [sorted(track_events.values(), key=lambda e: -e.score), sorted(windows, key=lambda e: -e.score)]
    for lst in ranked:
        for rank, ev in enumerate(lst):
            ev.rrf += 1.0 / (RRF_K + rank + 1)
    return sorted((e for lst in ranked for e in lst), key=lambda e: -e.rrf)[:limit]


PARTS = ("hand", "head", "torso", "legs", "feet")
SUBJECT_CLASSES = {"person": ("person",), "vehicle": ("car", "truck", "bus", "motorcycle", "bicycle"),
                   "bag": ("backpack", "handbag", "suitcase")}
MISSING_Z = -1.0       # no view of the part (too far away to show it): clearly below average on that condition
SIDE_PENALTY = 0.01    # a view of the other hand still counts, a little less (pose estimates swap sides)


@dataclass
class Query:
    """One condition to score: a visual phrase at the scale it is visible at."""
    phrase: str
    scale: str             # whole (whole-object crops) or a part: hand | head | torso | feet
    side: str | None       # L | R, the person's own, for hands
    vec: np.ndarray


def _person_box(conn: psycopg.Connection, track: int, ts: datetime) -> list[float] | None:
    t0, dts, xywh = conn.execute("select traj_t0, traj_dt_ms, traj_xywh from tracks where id = %s",
                                 (track,)).fetchone()
    target = (ts - t0).total_seconds() * 1000
    i = min(range(len(dts)), key=lambda j: abs(dts[j] - target))
    cx, cy, w, h = xywh[4 * i:4 * i + 4]
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def best_views(conn: psycopg.Connection, q: Query, classes: tuple[str, ...], cameras: list[str] | None = None,
               t0: datetime | None = None, t1: datetime | None = None) -> dict[int, tuple]:
    """Every track's best view for one condition, exact: track -> (similarity, view id, ts, box, side)."""
    hv = HalfVector(q.vec.astype(np.float16))
    where, args = _filters(cameras, t0, t1, "v")
    where = (where + " and " if where else "where ") + "t.class = any(%s)"
    if q.scale == "whole":
        sql = f"""select distinct on (v.track) v.track, v.id, v.ts, v.bbox, null::char, -(v.embedding <#> %s)
                  from track_crops v join tracks t on t.id = v.track {where}
                  order by v.track, v.embedding <#> %s"""
        params = [hv, *args, list(classes), hv]
    else:
        assert q.scale in PARTS
        sql = f"""select distinct on (v.track, v.side) v.track, v.id, v.ts, v.box, v.side, -(v.embedding <#> %s)
                  from part_crops v join tracks t on t.id = v.track {where} and v.part = %s
                  order by v.track, v.side, v.embedding <#> %s"""
        params = [hv, *args, list(classes), q.scale, hv]
    best: dict[int, tuple] = {}
    for track, vid, ts, box, side, sim in conn.execute(sql, params):
        sim -= SIDE_PENALTY if q.side and side != q.side else 0.0
        if track not in best or sim > best[track][0]:
            best[track] = (sim, vid, ts, box, side)
    return best


def search_attributes(conn: psycopg.Connection, queries: list[Query], subject: str, limit: int = 30,
                      cameras: list[str] | None = None, t0: datetime | None = None,
                      t1: datetime | None = None, missing_z: float | None = MISSING_Z,
                      close_ups_first: bool = True) -> list[Event]:
    """Tracked people or objects ranked by how well they meet every condition, each at its own scale.

    Each condition scores every track of the subject's classes exactly (its best view). Scores are
    standardized per condition, since one phrase's similarities don't compare with another's, and
    combined as mean + min of the z-scores: a track has to meet all conditions, not one very well.
    The moment is the best view of the most specific condition (a close-up shows the detail).
    missing_z: the score of a condition a track has no view for (too far away for that close-up); None
    leaves it out, for photo search, where every crop of the person is evidence but none is required.
    close_ups_first: the moment comes from the most specific condition; False takes the whole view."""
    classes = SUBJECT_CLASSES[subject]
    per = [best_views(conn, q, classes, cameras, t0, t1) for q in queries]
    zs = []
    for best in per:
        sims = np.array([v[0] for v in best.values()]) if best else np.zeros(1)
        mu, sd = float(sims.mean()), float(sims.std()) + 1e-6
        zs.append({t: (v[0] - mu) / sd for t, v in best.items()})

    def combined(t: int) -> float:
        z = [zi[t] if t in zi else missing_z for zi in zs]
        z = [v for v in z if v is not None]
        return float(np.mean(z) + min(z))

    ranked = sorted(set().union(*per), key=combined, reverse=True)[:limit]
    if not ranked:
        return []
    meta = {r[0]: r[1:] for r in conn.execute(
        "select id, camera, start_ts, end_ts, class from tracks where id = any(%s)", (ranked,))}
    # The moment and image: a close-up shows a detail best; a whole view shows a person best (photo search).
    focus = sorted(range(len(queries)), key=lambda i: (queries[i].scale == "whole") == close_ups_first)
    events = []
    for t in ranked:
        i = next(i for i in focus if t in per[i])
        sim, vid, ts, box, side = per[i][t]
        camera, start, end, cls = meta[t]
        ev = Event(camera, "track", start, end, ts, combined(t), None, t, cls, [vid])
        if queries[i].scale == "whole":
            ev.bbox = box
        else:
            ev.part, ev.part_box, ev.part_side = queries[i].scale, box, side
            ev.bbox = _person_box(conn, t, ts)
        ev.rrf = ev.score
        ev.condition_scores = [round(float(per[j][t][0]), 4) if t in per[j] else None for j in range(len(queries))]
        events.append(ev)
    return events


SAME_PLACE_IOU = 0.4
SAME_LOOK = 0.95        # mean crop embeddings: fragments of one object 0.97-0.99, two black SUVs ~0.83
MAX_OVERLAP_S = 1.0
TWIN_IOU, TWIN_LOOK = 0.8, 0.98


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def collapse(conn: psycopg.Connection, events: list[Event]) -> list[Event]:
    """Merge results that are one object tracked in pieces: a parked car or a seated person loses its
    track at every occlusion, and each piece would otherwise fill a slot. Pieces are merged when they
    are on one camera, in the same place, never on screen together, and look nearly identical; the best
    piece stays, its interval widens, and the rest are counted as sightings."""
    tracks = [e.track for e in events if e.kind == "track"]
    if len(tracks) < 2:
        return events
    crops: dict[int, list[np.ndarray]] = {}
    for track, emb in conn.execute("select track, embedding from track_crops where track = any(%s)", (tracks,)):
        crops.setdefault(track, []).append(emb.to_numpy().astype(np.float32))
    look = {}
    for t, vs in crops.items():
        mean = np.mean(vs, 0)
        look[t] = mean / np.linalg.norm(mean)
    place = {e.track: _person_box(conn, e.track, e.moment) for e in events if e.kind == "track"}
    kept: list[Event] = []
    pieces: dict[int, list[tuple[datetime, datetime]]] = {}   # kept track -> its pieces' own intervals
    apart = lambda a, b: min(a[1], b[1]) - max(a[0], b[0]) <= timedelta(seconds=MAX_OVERLAP_S)
    for e in events:
        if e.kind == "track" and e.track in look:
            for k in kept:
                if k.kind != "track" or k.camera != e.camera or k.track not in look:
                    continue
                iou, sim = _iou(place[k.track], place[e.track]), float(look[k.track] @ look[e.track])
                # Pieces in sequence; or two ids the tracker gave one object at once (same box, same look).
                if ((all(apart(p, (e.start, e.end)) for p in pieces[k.track]) and iou >= SAME_PLACE_IOU and sim >= SAME_LOOK)
                        or (iou >= TWIN_IOU and sim >= TWIN_LOOK)):
                    k.sightings.append(e.track)
                    pieces[k.track].append((e.start, e.end))
                    k.start, k.end = min(k.start, e.start), max(k.end, e.end)
                    break
            else:
                kept.append(e)
                pieces[e.track] = [(e.start, e.end)]
        else:
            kept.append(e)
    return kept
