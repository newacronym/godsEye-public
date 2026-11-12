"""Uploaded videos, end to end: upload -> prepare -> index on Modal -> load -> searchable.

    upload    the browser streams the file to the API (through the SSH tunnel, ~1 MB/s)
    prepare   FFmpeg on the Studio applies phone rotation and writes a 30 fps H.264 copy of at most 1080p
              (what gets indexed and cut into evidence), a 720p 15 fps copy for the browser, a thumbnail
    index     the 1080p copy goes to the Modal volume and index_upload runs both passes on an L4; its
              progress comes back through a Modal Dict
    load      the Parquet comes back and goes into Postgres, like the MEVA cameras

Each upload becomes a camera "U-xxxxxx" whose clock starts at the file's creation time (phones record
it), else the upload time. ~/data/uploads/<camera>/source.json is the record of it; the database rows
are derived and are rebuilt from ~/data/index/uploads/<camera>/*.parquet on a restart.
"""
from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

ROOT = Path.home() / "data/uploads"
INDEX = Path.home() / "data/index/uploads"
RUN = "uploads"
DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
APP, FUNCTION, VOLUME, PROGRESS = "godseye-indexer", "index_upload", "godseye-index", "godseye-progress"
DONE = ("ready", "failed")
PARQUET = ("tracks.parquet", "crops.parquet", "frames.parquet", "parts.parquet", "stats.json", "parts_stats.json")
MAX_LONG_SIDE = 1920
# Long side capped at 1920; the browser copy is 720p on the short side, like the MEVA viewing copies.
SCALE = "scale=w='if(gt(iw,ih),min({n},iw),-2)':h='if(gt(iw,ih),-2,min({n},ih))'"
PROXY_SCALE = "scale=w='if(gt(iw,ih),-2,min(720,iw))':h='if(gt(iw,ih),min(720,ih),-2)'"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Uploads:
    def __init__(self, on_ready: Callable[[], None] = lambda: None):
        ROOT.mkdir(parents=True, exist_ok=True)
        self.on_ready = on_ready
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="ingest")   # one video at a time
        for s in self.list():   # a restart interrupts whatever was in flight
            if s["status"] not in DONE:
                self._set(s["camera"], status="failed", message="Interrupted by a Studio restart; upload it again.")

    # ---- the record ----------------------------------------------------------------------------
    def list(self) -> list[dict]:
        out = []
        for f in sorted(ROOT.glob("*/source.json")):
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                continue
        return sorted(out, key=lambda s: s["created"], reverse=True)

    def get(self, camera: str) -> dict | None:
        f = ROOT / camera / "source.json"
        return json.loads(f.read_text()) if f.exists() else None

    def _set(self, cam: str, **fields) -> dict:
        with self.lock:
            s = self.get(cam) or {}
            s.update(fields, updated=_now())
            (ROOT / cam / "source.json").write_text(json.dumps(s, indent=2))
            return s

    # ---- upload --------------------------------------------------------------------------------
    def create(self, name: str, size: int | None) -> dict:
        camera = f"U-{secrets.token_hex(3)}"
        (ROOT / camera).mkdir(parents=True)
        return self._set(camera, camera=camera, name=name[:120] or "video", status="uploading", stage="uploading",
                         fraction=0.0, size=size, created=_now(), message=None)

    def writer(self, camera: str) -> "Writer":
        return Writer(self, camera)

    # ---- pipeline ------------------------------------------------------------------------------
    def _process(self, camera: str) -> None:
        try:
            meta = self._prepare(camera)
            self._index(camera, meta)
            self._load(camera)
            self._set(camera, status="ready", stage="ready", fraction=1.0, ready_at=_now())
            self.on_ready()
        except Exception as e:  # noqa: BLE001 - any failure is reported on the source, not raised into the worker
            traceback.print_exc()
            self._set(camera, status="failed", message=f"{type(e).__name__}: {e}"[:400])

    def _prepare(self, camera: str) -> dict:
        d = ROOT / camera
        raw = d / "upload.bin"
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "format=duration:format_tags=creation_time:stream=width,height", "-of", "json", str(raw)],
            capture_output=True, text=True, timeout=60).stdout or "{}")
        if not probe.get("streams"):
            raise ValueError("no video stream found; is this a video file?")
        duration = float(probe.get("format", {}).get("duration") or 0)
        created = probe.get("format", {}).get("tags", {}).get("creation_time")
        start = datetime.fromisoformat(created.replace("Z", "+00:00")) if created else datetime.now(timezone.utc)
        start = start.replace(microsecond=0)
        graph = (f"[0:v]fps=30,{SCALE.format(n=MAX_LONG_SIDE)},split=2[a][b];"
                 f"[b]fps=15,{PROXY_SCALE}[p]")
        cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-progress", "pipe:1", "-i", str(raw),
               "-filter_complex", graph,
               "-map", "[a]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-g", "30",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(d / "original.mp4"),
               "-map", "[p]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-b:v", "900k", "-maxrate", "1200k",
               "-bufsize", "1800k", "-g", "15", "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(d / "proxy.mp4")]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        last = 0.0
        for line in proc.stdout:
            if line.startswith("out_time_us=") and duration and time.monotonic() - last > 1:
                us = line.split("=", 1)[1].strip()
                if us.isdigit():
                    self._set(camera, fraction=round(min(1.0, int(us) / 1e6 / duration), 3))
                    last = time.monotonic()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.read()[-300:]}")
        out = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
             "stream=width,height,nb_read_packets", "-of", "json", str(d / "original.mp4")],
            capture_output=True, text=True, timeout=120).stdout)["streams"][0]
        frames = int(out["nb_read_packets"])
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{min(duration * 0.1, 5):.2f}",
                        "-i", str(d / "proxy.mp4"), "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4",
                        str(d / "thumb.jpg")], timeout=60)
        raw.unlink(missing_ok=True)
        end = start + timedelta(seconds=frames / 30)
        self._set(camera, start=start.isoformat(), end=end.isoformat(), duration_s=round(frames / 30, 1),
                  width=out["width"], height=out["height"], frames=frames, clock="file" if created else "upload")
        return {"start": start, "frames": frames}

    def _index(self, camera: str, meta: dict) -> None:
        import modal

        self._set(camera, status="indexing", stage="sending to the GPU", fraction=0.0)
        vol = modal.Volume.from_name(VOLUME)
        with vol.batch_upload(force=True) as batch:
            batch.put_file(str(ROOT / camera / "original.mp4"), f"uploads/{camera}/video.mp4")
        progress = modal.Dict.from_name(PROGRESS, create_if_missing=True)
        call = modal.Function.from_name(APP, FUNCTION).spawn(camera, meta["start"].isoformat(), meta["frames"])
        self._set(camera, stage="starting a GPU", modal_call=call.object_id)
        while True:
            try:
                stats = call.get(timeout=3)
                break
            except TimeoutError:
                p = progress.get(camera)
                if p:
                    self._set(camera, stage=p["stage"], fraction=p["fraction"])
        dest = INDEX / camera
        dest.mkdir(parents=True, exist_ok=True)
        self._set(camera, status="loading", stage="loading the index", fraction=0.0,
                  index_stats={k: stats.get(k) for k in ("video_s", "total_s", "realtime_x", "counts")})
        for name in PARQUET:
            with (dest / name).open("wb") as f:
                for chunk in vol.read_file(f"uploads/{camera}/{name}"):
                    f.write(chunk)

    def _load(self, camera: str) -> None:
        from indexer.load import load_camera

        with psycopg.connect(DSN) as conn:
            register_vector(conn)
            stats = load_camera(conn, RUN, INDEX / camera)
        self._set(camera, loaded=stats)

    # ---- delete --------------------------------------------------------------------------------
    def delete(self, camera: str) -> None:
        with psycopg.connect(DSN) as conn:
            conn.execute("delete from frame_embeds where camera = %s", (camera,))
            conn.execute("delete from tracks where camera = %s", (camera,))   # cascades to crops and parts
        shutil.rmtree(INDEX / camera, ignore_errors=True)
        shutil.rmtree(ROOT / camera, ignore_errors=True)
        try:
            import modal
            modal.Volume.from_name(VOLUME).remove_file(f"uploads/{camera}", recursive=True)
        except Exception:  # noqa: BLE001 - the volume copy is only a cache of the GPU input and output
            pass
        self.on_ready()


class Writer:
    """Takes the streamed upload body; closing it queues the rest of the pipeline."""

    def __init__(self, uploads: Uploads, camera: str):
        self.uploads, self.camera = uploads, camera
        self.size = (uploads.get(camera) or {}).get("size")
        self.f = (ROOT / camera / "upload.bin").open("wb")
        self.n, self.last = 0, 0.0

    def write(self, chunk: bytes) -> None:
        self.f.write(chunk)
        self.n += len(chunk)
        if self.size and time.monotonic() - self.last > 1:
            self.uploads._set(self.camera, fraction=round(self.n / self.size, 3))
            self.last = time.monotonic()

    def close(self) -> dict:
        self.f.close()
        if not self.n:
            return self.uploads._set(self.camera, status="failed", message="The upload was empty.")
        self.uploads._set(self.camera, status="preparing", stage="preparing the video", fraction=0.0, bytes=self.n)
        self.uploads.worker.submit(self.uploads._process, self.camera)
        return self.uploads.get(self.camera)
