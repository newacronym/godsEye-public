"""Batch indexing on Modal: one L4 container per camera, Parquet out to a Modal Volume.

    modal run services/indexer/index_app.py --cameras G330            # one camera (smoke test)
    modal run services/indexer/index_app.py                            # whole spike slice
    modal run services/indexer/index_app.py --parts                    # body-part pass over existing tracks
    modal volume get godseye-index spike ~/data/index/                 # then fetch on the Studio
    modal deploy services/indexer/index_app.py                         # makes index_upload callable from the Studio

The Studio loads the Parquet into Postgres (services/indexer/load.py).
"""
import json
import sys
from pathlib import Path

import modal

YOLO_WEIGHTS = "yolo26m.pt"
POSE_WEIGHTS = "yolo26l-pose.pt"
SIGLIP = "google/siglip2-so400m-patch16-384"
MODELS = "/models"


def download_models() -> None:
    import os
    from transformers import AutoModel, AutoProcessor
    from ultralytics import YOLO

    os.makedirs(MODELS, exist_ok=True)
    os.chdir(MODELS)
    YOLO(YOLO_WEIGHTS)
    YOLO(POSE_WEIGHTS)
    AutoProcessor.from_pretrained(SIGLIP, cache_dir=f"{MODELS}/hf")
    AutoModel.from_pretrained(SIGLIP, cache_dir=f"{MODELS}/hf")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.14.0", "torchvision==0.29.0", "ultralytics==8.4.163", "transformers==5.17.0",
        "supervision==0.30.5", "av==18.1.0", "pyarrow==25.0.1", "numpy", "pillow",
        "fastapi[standard]==0.141.1",   # the live indexer's websocket endpoint
    )
    .run_function(download_models)
    .env({"HF_HUB_OFFLINE": "1", "YOLO_OFFLINE": "1"})  # weights are baked in; no downloads at runtime
    .add_local_python_source("pipeline", "parts", "geometry", "live")
)
app = modal.App("godseye-indexer", image=image)
volume = modal.Volume.from_name("godseye-index", create_if_missing=True)
progress = modal.Dict.from_name("godseye-progress", create_if_missing=True)   # upload id -> {stage, fraction}


@app.function(gpu="L4", cpu=8, memory=16384, volumes={"/index": volume}, timeout=3600)
def index_camera(run: str, camera: str, clips: list[dict]) -> dict:
    import tempfile
    import time

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    from pipeline import SRC_FPS, CameraIndexer, Embedder

    LOGGER.setLevel("ERROR")  # silence per-batch deprecation warnings

    t0 = time.perf_counter()
    indexer = CameraIndexer(camera, YOLO(f"{MODELS}/{YOLO_WEIGHTS}"), Embedder(SIGLIP, f"{MODELS}/hf"))
    load_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        indexer.run(clips, Path(tmp))
    total = time.perf_counter() - t0
    stats = indexer.write(Path(f"/index/{run}/{camera}"), indexer.counts["frames_decoded"] / SRC_FPS, total)
    stats["timing_s"]["model_load_s"] = round(load_s, 1)
    volume.commit()
    return stats


@app.function(gpu="L4", cpu=8, memory=16384, volumes={"/index": volume}, timeout=3600)
def index_parts(run: str, camera: str, clips: list[dict]) -> dict:
    """Second pass: body-part crops for the tracked people of an already indexed camera."""
    import tempfile
    import time

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    from parts import PartIndexer, Persons
    from pipeline import SRC_FPS, Embedder

    LOGGER.setLevel("ERROR")

    t0 = time.perf_counter()
    out = Path(f"/index/{run}/{camera}")
    indexer = PartIndexer(camera, YOLO(f"{MODELS}/{POSE_WEIGHTS}"), Embedder(SIGLIP, f"{MODELS}/hf"),
                          Persons(out / "tracks.parquet"))
    load_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        indexer.run(clips, Path(tmp))
    stats = indexer.write(out, indexer.counts["frames_decoded"] / SRC_FPS, time.perf_counter() - t0)
    stats["timing_s"]["model_load_s"] = round(load_s, 1)
    volume.commit()
    return stats


@app.function(gpu="L4", cpu=8, memory=16384, volumes={"/index": volume}, timeout=3600)
def index_upload(camera: str, start: str, frames: int) -> dict:
    """An uploaded video, both passes in one container. The Studio has already normalized it to 30 fps
    H.264 and put it at /index/uploads/<camera>/video.mp4; Parquet goes next to it. Progress goes to
    the godseye-progress Dict so the console can show it."""
    import tempfile
    import threading
    import time

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    from parts import PartIndexer, Persons
    from pipeline import SRC_FPS, CameraIndexer, Embedder

    LOGGER.setLevel("ERROR")
    volume.reload()
    out = Path(f"/index/uploads/{camera}")
    clip = {"file": "video.mp4", "path": str(out / "video.mp4"), "start": start}
    t0 = time.perf_counter()
    progress[camera] = {"stage": "loading models", "fraction": 0.0}
    embedder = Embedder(SIGLIP, f"{MODELS}/hf")
    stages = [("finding and tracking people and vehicles", 0.0, 0.6), ("close-ups of hands, heads, torsos, feet", 0.6, 1.0)]
    current: dict = {}

    def report() -> None:
        while not current.get("done"):
            ix = current.get("indexer")
            if ix is not None:
                name, lo, hi = stages[current["stage"]]
                frac = min(1.0, ix.counts["frames_decoded"] / max(frames, 1))
                progress[camera] = {"stage": name, "fraction": round(lo + (hi - lo) * frac, 3)}
            time.sleep(2)

    threading.Thread(target=report, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            first = CameraIndexer(camera, YOLO(f"{MODELS}/{YOLO_WEIGHTS}"), embedder)
            current.update(indexer=first, stage=0)
            first.run([clip], Path(tmp))
            video_s = first.counts["frames_decoded"] / SRC_FPS
            stats = first.write(out, video_s, time.perf_counter() - t0)
            second = PartIndexer(camera, YOLO(f"{MODELS}/{POSE_WEIGHTS}"), embedder, Persons(out / "tracks.parquet"))
            current.update(indexer=second, stage=1)
            t1 = time.perf_counter()
            second.run([clip], Path(tmp))
            stats["parts"] = second.write(out, video_s, time.perf_counter() - t1)["counts"]
    finally:
        current["done"] = True
    volume.commit()
    progress[camera] = {"stage": "done", "fraction": 1.0}
    return stats


LIVE_BATCH = 5           # frames per GPU step (~1 s of a 5 fps stream)


@app.cls(gpu="L4", cpu=4, memory=16384, timeout=4 * 3600, scaledown_window=60,
         secrets=[modal.Secret.from_name("godseye-live")])
@modal.concurrent(max_inputs=1)   # one stream per GPU container; a second stream gets its own
class Live:
    """Live indexing over a websocket. The Studio connects to /ws/<camera> with the shared
    token, sends 8-byte little-endian wall-clock seconds + a JPEG per frame, and gets a JSON message of
    new rows after every GPU step. The container stops a minute after the last stream disconnects."""

    @modal.enter()
    def load(self) -> None:
        import numpy as np
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        from pipeline import Embedder

        LOGGER.setLevel("ERROR")
        self.detector = YOLO(f"{MODELS}/{YOLO_WEIGHTS}")
        self.pose = YOLO(f"{MODELS}/{POSE_WEIGHTS}")
        self.embedder = Embedder(SIGLIP, f"{MODELS}/hf")
        blank = np.zeros((720, 1280, 3), np.uint8)   # first inference compiles kernels; do it before a stream waits
        self.detector.predict([blank], imgsz=1280, half=True, verbose=False)
        self.pose.predict([blank], imgsz=512, half=True, verbose=False)

    @modal.asgi_app()
    def web(self):
        import asyncio
        import hmac
        import os
        import struct
        import time

        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        from live import LiveIndexer

        api = FastAPI()

        @api.websocket("/ws/{camera}")
        async def stream(socket: WebSocket, camera: str) -> None:
            token = socket.headers.get("authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(token, os.environ["LIVE_TOKEN"]):
                await socket.close(code=4401)
                return
            await socket.accept()
            indexer = LiveIndexer(camera, self.detector, self.pose, self.embedder)
            await socket.send_json({"type": "ready"})
            frames, last = [], time.monotonic()
            try:
                while True:
                    msg = await socket.receive_bytes()
                    frames.append((struct.unpack("<d", msg[:8])[0], msg[8:]))
                    if len(frames) >= LIVE_BATCH or time.monotonic() - last >= 1.0:
                        rows = await asyncio.to_thread(indexer.process, frames)   # GPU work off the event loop
                        await socket.send_json({"type": "rows", "received": len(frames)} | rows)
                        frames, last = [], time.monotonic()
            except WebSocketDisconnect:
                pass

        return api


@app.local_entrypoint()
def main(date: str = "2018-03-11", start: str = "16:15", end: str = "16:45", site: str = "school",
         cameras: str = "", run: str = "spike", parts: bool = False) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "meva"))
    from download import list_clips

    picked = list_clips(date, start, end, site, set(cameras.split(",")) if cameras else None)
    by_cam: dict[str, list[dict]] = {}
    for name, (key, size, m) in sorted(picked.items()):
        by_cam.setdefault(m["cam"], []).append({
            "file": name, "key": key, "bytes": size,
            "start": f"{m['date']}T{m['start'].replace('-', ':')}",
        })
    fn = index_parts if parts else index_camera
    print(f"{"parts" if parts else "index"}: {sum(map(len, by_cam.values()))} clips from {len(by_cam)} cameras -> volume godseye-index/{run}")
    for stats in fn.starmap([(run, cam, clips) for cam, clips in by_cam.items()]):
        print(json.dumps(stats))
