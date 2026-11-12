"""HTTP API on the Studio, reached from the Mac at http://localhost:8000 through `make tunnel`.

    GET /health                    footage summary
    GET /cameras                   cameras with footage range, track counts and live stream names
    GET /search?q=...              server-sent events: provisional, plan, matched, verdict..., done
    GET /media/crop/{id}.jpg       stored crop of a tracked object
    GET /media/tile/{id}.jpg       whole-scene thumbnail nearest a frame-tile hit
    GET /media/part/{id}.jpg       native-resolution view of a body part (hand, head, torso)
    GET /events/{id}/details?part= every stored view of that body part of a track event
    GET /playback?camera=&ts=      720p proxy clip URL + offset for a moment
    GET /footage/{camera}          a camera's or upload's viewing clips in time order (to watch it)
    GET /footage/{camera}/thumb.jpg  a scene frame from early in that footage
    GET /events/{id}/boxes         every tracked box of a track event (the player draws it over time)
    GET /live                      live streams in MediaMTX (fake-live MEVA cameras, the phone), with
                                   their live-indexing session if one runs
    POST/DELETE /live/{path}/index start / stop indexing a live stream on a Modal GPU
    GET /live-video?camera=&start=&duration=  a window of a live stream's recording (MP4)
    GET /video/<file>.mp4          the proxy clips (range requests, for the player)
    POST /sources?name=            upload a video (raw body); it is prepared, indexed on Modal, loaded
    GET /sources                   uploaded videos with their status and progress
    DELETE /sources/{camera}       remove an uploaded video and its index
    GET /uploads/<camera>/...      an upload's 720p copy (proxy.mp4) and thumbnail (thumb.jpg)
    POST /photos                   a photo to search by (raw body): returns its id and the people found in it
    GET /photos/{id}/{person}.jpg  the chosen person's crop
    GET /search?photo=&person=     photo search (q adds conditions in words)
"""
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from ingest.uploads import ROOT as UPLOADS
from ingest.uploads import Uploads
from live.session import Session as LiveSession
from search import service
from search.encoder import ImageEncoder, TextEncoder
from search.photo import Photos
from search.evidence import footage as clips, track_points
from search.llm import Claude

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
PROXY = Path.home() / "data/meva/proxy"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = ConnectionPool(DSN, min_size=2, max_size=16, configure=register_vector, open=True)

    def refresh() -> None:   # after an upload is loaded or deleted
        clips().refresh()
        state["footage"] = service.footage(pool, {s["camera"]: s["name"] for s in state["uploads"].list()})

    state.update(pool=pool, enc=TextEncoder(), claude=Claude(), uploads=Uploads(on_ready=refresh), photos=Photos(),
                 live={}, refresh=refresh)
    refresh()
    # The image tower (photo search, ~1.6 GB) and the photo pose model load in the background, so the
    # API is up at once and the first photo search doesn't wait for them.
    threading.Thread(target=lambda: (image_encoder(), state["photos"].pose()), daemon=True).start()
    yield
    for session in state["live"].values():
        session.stop()
    pool.close()


app = FastAPI(title="godsEye", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["GET", "POST", "DELETE"],
                   allow_headers=["*"])
app.mount("/video", StaticFiles(directory=PROXY), name="video")
UPLOADS.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")


def _utc(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt and dt.tzinfo is None else dt


@app.get("/health")
def health() -> dict:
    return {"ok": True, "footage": state["footage"].description}


@app.get("/cameras")
def cameras() -> list[dict]:
    with state["pool"].connection() as conn:
        rows = conn.execute("""select camera, min(start_ts), max(end_ts), count(*), bool_or(run like 'live-%%')
                               from tracks group by camera order by camera""").fetchall()
    return [{"id": cam, "start": s.isoformat(), "end": e.isoformat(), "tracks": n, "live": f"meva-{cam}",
             "kind": "live" if live else "upload" if cam.startswith("U-") else "library"}
            for cam, s, e, n, live in rows]


_img_lock = threading.Lock()


def image_encoder() -> ImageEncoder:
    with _img_lock:
        if "img_enc" not in state:
            state["img_enc"] = ImageEncoder()
        return state["img_enc"]


@app.get("/search")
def search(q: str = "", cameras: str | None = None, t0: datetime | None = None, t1: datetime | None = None,
           verify_top: int = 20, photo: str | None = None, person: int = 0) -> StreamingResponse:
    if not q.strip() and not photo:
        raise HTTPException(400, "empty query")
    req = service.Request(q=q.strip(), cameras=cameras.split(",") if cameras else None, t0=_utc(t0), t1=_utc(t1),
                          verify_top=max(0, min(verify_top, 40)), photo=photo, person=person)

    def stream():
        events = (service.run_photo(req, state["pool"], state["enc"], image_encoder(), state["photos"],
                                    state["claude"], state["footage"]) if photo else
                  service.run(req, state["pool"], state["enc"], state["claude"], state["footage"]))
        for event in events:
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _jpeg(row) -> Response:
    if not row or not row[0]:
        raise HTTPException(404)
    return Response(bytes(row[0]), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/media/crop/{crop_id}.jpg")
def crop(crop_id: int) -> Response:
    with state["pool"].connection() as conn:
        return _jpeg(conn.execute("select jpeg from track_crops where id = %s", (crop_id,)).fetchone())


@app.get("/media/part/{part_id}.jpg")
def part_crop(part_id: int) -> Response:
    with state["pool"].connection() as conn:
        return _jpeg(conn.execute("select jpeg from part_crops where id = %s", (part_id,)).fetchone())


@app.get("/events/{event_id}/details")
def event_details(event_id: str, part: str) -> list[dict]:
    """Every stored view of a person's body part (hand, head or torso), in time order."""
    if not event_id.startswith("t") or not event_id[1:].isdigit():
        return []
    with state["pool"].connection() as conn:
        rows = conn.execute("""select id, ts, side, box, px from part_crops where track = %s and part = %s
                               order by ts""", (int(event_id[1:]), part)).fetchall()
    return [{"id": i, "ts": ts.isoformat(), "side": side, "box": [round(v, 4) for v in box], "px": px,
             "image": f"/media/part/{i}.jpg"} for i, ts, side, box, px in rows]


@app.get("/media/tile/{frame_id}.jpg")
def tile(frame_id: int) -> Response:
    with state["pool"].connection() as conn:
        return _jpeg(conn.execute(
            """select t.thumb from frame_embeds f
               join lateral (select thumb from frame_embeds t where t.camera = f.camera and t.tile = 0
                             and t.thumb is not null order by abs(extract(epoch from t.ts - f.ts)) limit 1) t on true
               where f.id = %s""", (frame_id,)).fetchone())


def _viewing_copy(path: Path) -> tuple[Path, str]:
    """The 720p copy the browser plays for an original clip, and its URL."""
    if path.name == "original.mp4":   # an upload: its viewing copy sits next to it
        return path.parent / "proxy.mp4", f"/uploads/{path.parent.name}/proxy.mp4"
    return PROXY / (path.stem + ".mp4"), f"/video/{path.stem}.mp4"


@app.get("/playback")
def playback(camera: str, ts: datetime) -> dict:
    loc = clips().locate(camera, _utc(ts))
    if loc is None:
        raise HTTPException(404, "no footage at that time")
    path, offset, duration = loc
    clip_start = _utc(ts) - timedelta(seconds=offset)
    if isinstance(path, str):   # a live stream's recording, cut by MediaMTX
        return {"url": f"/live-video?camera={camera}&start={clip_start.isoformat().replace('+00:00', 'Z')}"
                       f"&duration={duration:g}", "offset": round(offset, 2), "clip_start": clip_start.isoformat(),
                "duration": duration}
    proxy, url = _viewing_copy(path)
    if not proxy.exists():
        raise HTTPException(404, "no viewing copy for that clip")
    return {"url": url, "offset": round(offset, 2), "clip_start": clip_start.isoformat(), "duration": duration}


@app.get("/footage/{camera:path}/thumb.jpg")
def footage_thumb(camera: str) -> Response:
    """A whole-scene frame from early in the footage (stored at indexing time)."""
    with state["pool"].connection() as conn:
        return _jpeg(conn.execute("""select thumb from frame_embeds where camera = %s and tile = 0
                                     and thumb is not null order by ts offset 3 limit 1""", (camera,)).fetchone())


@app.get("/footage/{camera:path}")
def footage(camera: str) -> dict:
    """A camera's or an upload's clips in time order, for watching it from start to end."""
    found = clips().clips.get(camera)
    if not found:
        raise HTTPException(404, "no footage for that camera")
    upload = state["uploads"].get(camera) if camera.startswith("U-") else None
    out = []
    for start, end, path in found:
        proxy, url = _viewing_copy(path)
        if proxy.exists():
            out.append({"start": start.isoformat(), "end": end.isoformat(), "url": url})
    return {"camera": camera, "name": upload["name"] if upload else f"Camera {camera}",
            "kind": "upload" if upload else "library", "clips": out}


@app.post("/photos")
async def add_photo(request: Request) -> dict:
    data = await request.body()
    if not data or len(data) > 20_000_000:
        raise HTTPException(400, "send one image, under 20 MB")
    try:
        return await run_in_threadpool(state["photos"].add, data)   # the pose model takes ~0.5 s on CPU
    except OSError as e:   # PIL can't read it
        raise HTTPException(400, f"not an image: {e}") from e


@app.get("/photos/{photo}/{person}.jpg")
def photo_person(photo: str, person: int) -> Response:
    try:
        return Response(state["photos"].jpeg(photo, person), media_type="image/jpeg")
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, "no such photo") from e


@app.post("/sources")
async def upload_source(request: Request, name: str = "video") -> dict:
    """The raw request body is the video file; it streams to disk, then the pipeline runs in the background."""
    up = state["uploads"]
    src = up.create(name, int(request.headers.get("content-length") or 0) or None)
    writer = up.writer(src["camera"])
    async for chunk in request.stream():
        writer.write(chunk)
    return writer.close()


@app.get("/sources")
def sources() -> dict:
    with state["pool"].connection() as conn:
        meva = conn.execute("""select count(distinct camera), min(start_ts), max(end_ts), count(*) from tracks
                               where camera not like 'U-%' and run not like 'live-%'""").fetchone()
    return {"uploads": state["uploads"].list(),
            "library": {"name": "MEVA school site", "cameras": meva[0], "tracks": meva[3],
                        "start": meva[1].isoformat() if meva[1] else None, "end": meva[2].isoformat() if meva[2] else None}}


@app.delete("/sources/{camera}")
def delete_source(camera: str) -> dict:
    if not camera.startswith("U-") or state["uploads"].get(camera) is None:
        raise HTTPException(404, "no such upload")
    state["uploads"].delete(camera)
    return {"deleted": camera}


@app.get("/events/{event_id}/boxes")
def event_boxes(event_id: str) -> list[dict]:
    if not event_id.startswith("t") or not event_id[1:].isdigit():
        return []
    with state["pool"].connection() as conn:
        pts = track_points(conn, int(event_id[1:]))
    return [{"ts": t.isoformat(), "box": [round(v, 4) for v in box]} for t, box in pts]


@app.get("/live")
def live() -> list[dict]:
    with urllib.request.urlopen("http://127.0.0.1:9997/v3/paths/list", timeout=3) as r:
        items = json.load(r)["items"]
    sessions = state["live"]
    return [{"name": p["name"], "ready": p["ready"], "readers": len(p.get("readers", [])),
             "indexing": sessions[p["name"]].status() if p["name"] in sessions else None}
            for p in sorted(items, key=lambda p: p["name"])]


@app.post("/live/{path:path}/index")    # stream names can nest: live/phone1
def start_live(path: str) -> dict:
    """Index a live stream as it arrives, on a Modal L4 (about $0.80 an hour while it runs)."""
    current = state["live"].get(path)
    if current and current.state in ("starting the GPU", "indexing"):
        return current.status()
    if not any(p["name"] == path and p["ready"] for p in live()):
        raise HTTPException(404, "no live stream by that name")
    state["live"][path] = LiveSession(path, on_change=state["refresh"])
    return state["live"][path].status()


@app.delete("/live/{path:path}/index")
def stop_live(path: str) -> dict:
    session = state["live"].get(path)
    if session is None:
        raise HTTPException(404, "not indexing that stream")
    session.stop()
    return session.status()


@app.get("/live-video")
def live_video(camera: str, start: str, duration: float = 30.0) -> StreamingResponse:
    """A window of a live stream's recording, from MediaMTX's playback server (fragmented MP4)."""
    if ".." in camera or camera.startswith("/") or duration > 300:
        raise HTTPException(400, "bad request")
    url = "http://127.0.0.1:9996/get?" + urllib.parse.urlencode({"path": camera, "start": start, "duration": duration})
    try:
        upstream = urllib.request.urlopen(url, timeout=15)
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, "no recording for that window") from e
    return StreamingResponse(iter(lambda: upstream.read(1 << 16), b""), media_type="video/mp4")
