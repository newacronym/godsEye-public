"""The search pipeline, shared by the API and the evaluation so what we measure ships.

Stages, each yielded as an event as soon as it is ready:
  provisional  raw query -> SigLIP 2 -> kNN; only filters the user set explicitly are applied
  plan         Claude's interpretation: conditions with their scales, intent, cameras, time window
               (runs in parallel with the provisional search)
  matched      for how something looks, every tracked person or object scored on each condition at
               its own scale (whole object, or hand, head, torso, feet close-ups); for activities and
               untracked things, the scene-level search with the plan's phrase and filters
  verdict      one per checked candidate, as each finishes: yes / no / not visible per condition,
               and match, possible or no (evidence and Claude calls run in parallel)
  done         timings
"""
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone

from psycopg_pool import ConnectionPool

from search.encoder import ImageEncoder, TextEncoder
from search.evidence import gather
from search.llm import APPEARANCE_MODEL, MODEL, MODEL_NAMES, Claude, Plan
from search.photo import Photos
from search.retrieve import Event, Query, collapse, search, search_attributes

EVIDENCE_WORKERS = 4    # ffmpeg strips are CPU-bound: one per core
VERIFY_WORKERS = 10


@dataclass
class Request:
    q: str
    cameras: list[str] | None = None
    t0: datetime | None = None
    t1: datetime | None = None
    limit: int = 30
    verify_top: int = 20
    photo: str | None = None   # photo search: an uploaded photo's id, and which person in it
    person: int = 0


@dataclass
class Footage:
    """What the planner is told about the indexed footage."""
    description: str
    cameras: list[str]
    start: datetime
    end: datetime


def event_id(e: Event) -> str:
    return f"t{e.track}" if e.kind == "track" else f"w{e.evidence[0]}"


def event_json(e: Event) -> dict:
    out = {"id": event_id(e), "kind": e.kind, "camera": e.camera, "class": e.cls,
           "start": e.start.isoformat(), "end": e.end.isoformat(), "moment": e.moment.isoformat(),
           "score": round(float(e.score), 4), "bbox": [round(float(v), 4) for v in e.bbox],
           "image": f"/media/crop/{e.evidence[0]}.jpg" if e.kind == "track" else f"/media/tile/{e.evidence[0]}.jpg"}
    if e.part:
        out["image"] = f"/media/part/{e.evidence[0]}.jpg"
        out["detail"] = {"part": e.part, "side": e.part_side, "box": [round(float(v), 4) for v in e.part_box]}
    if e.condition_scores:
        out["condition_scores"] = e.condition_scores
    if e.sightings:
        out["sightings"] = 1 + len(e.sightings)
    return out


def footage(pool: ConnectionPool, names: dict[str, str] | None = None) -> Footage:
    """What is indexed, described for the planner: the MEVA cameras, and each uploaded video by name."""
    with pool.connection() as conn:
        rows = conn.execute("""select camera, min(start_ts), max(end_ts), bool_or(run like 'live-%%') from tracks
                               group by 1 order by 1""").fetchall()
    if not rows:
        now = datetime.now(timezone.utc)
        return Footage("nothing indexed yet", [], now, now)
    live = [r for r in rows if r[3]]
    meva = [r for r in rows if not r[3] and not r[0].startswith("U-")]
    uploads = [r for r in rows if not r[3] and r[0].startswith("U-")]
    parts = []
    if meva:
        parts.append(f"{len(meva)} MEVA cameras ({', '.join(r[0] for r in meva)}), "
                     f"{min(r[1] for r in meva):%Y-%m-%d %H:%M}-{max(r[2] for r in meva):%H:%M} UTC")
    for cam, start, end, _ in uploads:
        parts.append(f"uploaded video {cam} '{(names or {}).get(cam, cam)}' ({start:%Y-%m-%d %H:%M} UTC, "
                     f"{(end - start).total_seconds() / 60:.0f} min)")
    for cam, start, end, _ in live:
        parts.append(f"live camera {cam} (indexed live since {start:%Y-%m-%d %H:%M} UTC, latest {end:%H:%M:%S} UTC)")
    return Footage("; ".join(parts), [r[0] for r in rows], min(r[1] for r in rows), max(r[2] for r in rows))


def _plan_window(plan: Plan, f: Footage) -> tuple[datetime | None, datetime | None]:
    def at(hhmm: str | None) -> datetime | None:
        if not hhmm:
            return None
        h, m = map(int, hhmm.split(":")[:2])
        return f.start.replace(hour=h, minute=m, second=0, microsecond=0)
    return at(plan.time_from), at(plan.time_to)


def _queries(plan: Plan | None, enc: TextEncoder) -> list[Query]:
    attrs = plan.useful_attributes() if plan else []
    if not attrs:
        return []
    vecs = enc.encode([a.phrase for a in attrs])
    side = {"left": "L", "right": "R"}
    return [Query(a.phrase, a.scale, side.get(a.side) if a.scale == "hand" else None, v)
            for a, v in zip(attrs, vecs)]


def run(req: Request, pool: ConnectionPool, enc: TextEncoder, claude: Claude, f: Footage) -> Iterator[dict]:
    t_start = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t_start) * 1000)

    with ThreadPoolExecutor(1) as ex:
        plan_future = ex.submit(claude.plan, req.q, f.description, f.cameras)
        vec = enc.encode([req.q])[0]
        with pool.connection() as conn:
            provisional = search(conn, vec, limit=req.limit, cameras=req.cameras, t0=req.t0, t1=req.t1)
        yield {"type": "provisional", "ms": ms(), "events": [event_json(e) for e in provisional]}
        plan = plan_future.result()
    yield {"type": "plan", "ms": ms(), "plan": plan.model_dump() if plan else None}

    cameras, t0, t1, phrase = req.cameras, req.t0, req.t1, req.q
    if plan:
        cameras = cameras or [c for c in plan.cameras if c in f.cameras] or None
        p0, p1 = _plan_window(plan, f)
        t0, t1 = t0 or p0, t1 or p1
        phrase = plan.object_phrase or req.q
    intent = plan.intent if plan and plan.intent != "count" else "appearance"
    conditions = (plan.conditions() if plan else []) or [req.q]
    queries = _queries(plan, enc)
    parts = [q for q in queries if q.scale != "whole"]

    # How things look: score every tracked person or object on each condition at its own scale.
    # Activities, and things the detector has no class for, use the scene-level search.
    matched, mode = None, "scene"
    if queries and plan.subject != "other" and (intent == "appearance" or parts):
        with pool.connection() as conn:
            matched = search_attributes(conn, queries, plan.subject, limit=3 * req.limit, cameras=cameras, t0=t0, t1=t1)
        mode = "attributes" if matched else mode
    if not matched:
        vec = enc.encode([phrase])[0] if phrase.strip().lower() != req.q.strip().lower() else vec
        with pool.connection() as conn:
            matched = search(conn, vec, limit=2 * req.limit, cameras=cameras, t0=t0, t1=t1)
    with pool.connection() as conn:     # one parked car tracked in 17 pieces is one result
        matched = collapse(conn, matched)[:req.limit]
    yield {"type": "matched", "ms": ms(), "mode": mode, "phrase": phrase, "conditions": conditions,
           "attributes": [a.model_dump() for a in plan.useful_attributes()] if plan else [], "cameras": cameras,
           "t0": t0.isoformat() if t0 else None, "t1": t1.isoformat() if t1 else None,
           "events": [event_json(e) for e in matched]}

    model = APPEARANCE_MODEL if intent == "appearance" or parts else MODEL
    key = plan.key_conditions() if plan else [True]
    yield from _verify(req.q, matched[:req.verify_top], conditions, intent, parts, model, pool, claude, ms, key=key)
    yield {"type": "done", "ms": ms(), "intent": intent, "mode": mode, "verified": min(len(matched), req.verify_top)}


def _verify(question: str, cands: list[Event], conditions: list[str], intent: str, parts: list[Query], model: str,
            pool: ConnectionPool, claude: Claude, ms, reference: tuple[str, bytes] | None = None,
            key: list[bool] | None = None) -> Iterator[dict]:
    """Evidence and verification overlap: each candidate goes to Claude as soon as its evidence is ready,
    and each verdict streams out as soon as it lands. A reference image (photo search) leads every set."""
    def evidence(e: Event) -> list[tuple[str, bytes]]:
        with pool.connection() as conn:
            return ([reference] if reference else []) + gather(conn, e, intent, parts)

    with ThreadPoolExecutor(EVIDENCE_WORKERS) as ev_ex, ThreadPoolExecutor(VERIFY_WORKERS) as v_ex:
        pending = {ev_ex.submit(evidence, e): ("evidence", e) for e in cands}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                stage, e = pending.pop(fut)
                if fut.exception():
                    yield {"type": "verdict", "ms": ms(), "id": event_id(e), "match": None, "unsure": None,
                           "confidence": None, "checks": [],
                           "reason": f"{stage} failed: {type(fut.exception()).__name__}"}
                elif stage == "evidence":
                    pending[v_ex.submit(claude.verify, question, conditions, fut.result(), intent, model,
                                        reference is not None, key)] = ("verify", e)
                else:
                    v = fut.result()
                    yield {"type": "verdict", "ms": ms(), "id": event_id(e),
                           "match": v.match if v else None, "unsure": v.unsure if v else None,
                           "confidence": v.confidence if v else None, "checks": v.checks if v else [],
                           "reason": v.reason if v else "verifier unavailable", "by": MODEL_NAMES[model]}


SAME_PERSON = "the same person as in the reference photo"


def run_photo(req: Request, pool: ConnectionPool, enc: TextEncoder, img_enc: ImageEncoder, photos: Photos,
              claude: Claude, f: Footage) -> Iterator[dict]:
    """Find the person in a photo: each crop of them (whole body, head, torso, legs, feet) is a condition
    scored on every tracked person; text in the question adds its own conditions; Sonnet compares the
    photo with each top candidate."""
    t_start = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t_start) * 1000)
    with ThreadPoolExecutor(1) as ex:
        plan_future = ex.submit(claude.plan, req.q, f.description, f.cameras) if req.q.strip() else None
        crops = photos.crops(req.photo, req.person)
        vecs = dict(zip(crops, img_enc.encode(list(crops.values()))))
        with pool.connection() as conn:
            provisional = search(conn, vecs["whole"], limit=req.limit, cameras=req.cameras, t0=req.t0, t1=req.t1)
        yield {"type": "provisional", "ms": ms(), "events": [event_json(e) for e in provisional]}
        plan = plan_future.result() if plan_future else None

    shown = {"whole": "the person in your photo"} | {p: f"their {p}" for p in crops if p != "whole"}
    photo_attrs = [{"phrase": shown[p], "scale": p, "side": "either"} for p in crops]
    text_attrs = [a.model_dump() for a in plan.useful_attributes()] if plan else []
    yield {"type": "plan", "ms": ms(), "plan": {
        "subject": "person", "object_phrase": "the person in your photo" + (f", {req.q}" if req.q.strip() else ""),
        "attributes": photo_attrs + text_attrs, "activity": plan.activity if plan else "",
        "intent": "appearance", "cameras": plan.cameras if plan else [],
        "time_from": plan.time_from if plan else None, "time_to": plan.time_to if plan else None}}

    cameras, t0, t1 = req.cameras, req.t0, req.t1
    if plan:
        cameras = cameras or [c for c in plan.cameras if c in f.cameras] or None
        p0, p1 = _plan_window(plan, f)
        t0, t1 = t0 or p0, t1 or p1
    queries = [Query(shown[p], p, None, v) for p, v in vecs.items()] + _queries(plan, enc)
    with pool.connection() as conn:
        matched = search_attributes(conn, queries, "person", limit=3 * req.limit, cameras=cameras, t0=t0, t1=t1,
                                    missing_z=None,   # far-away people have no close-ups; that is not evidence against
                                    close_ups_first=False)
        matched = collapse(conn, matched)[:req.limit]
    conditions = [SAME_PERSON] + (plan.conditions() if plan else [])
    yield {"type": "matched", "ms": ms(), "mode": "photo", "phrase": "the person in your photo",
           "conditions": conditions, "attributes": photo_attrs + text_attrs, "cameras": cameras,
           "t0": t0.isoformat() if t0 else None, "t1": t1.isoformat() if t1 else None,
           "events": [event_json(e) for e in matched]}

    reference = ("Reference photo of the person to find:", photos.jpeg(req.photo, req.person))
    parts = [q for q in queries if q.scale != "whole"]
    question = req.q.strip() or "Is this the person in the reference photo?"
    key = [True] + (plan.key_conditions() if plan else [])     # the person in the photo is the point
    yield from _verify(question, matched[:req.verify_top], conditions, "appearance", parts, APPEARANCE_MODEL,
                       pool, claude, ms, reference, key)
    yield {"type": "done", "ms": ms(), "intent": "appearance", "mode": "photo", "verified": min(len(matched), req.verify_top)}
