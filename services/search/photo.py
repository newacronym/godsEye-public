"""Photo search: find the person in an uploaded photo across the footage.

    photo -> pose model (CPU) -> the people in it, largest first; the user can pick another
          -> crops of the chosen person: whole body, head, torso, legs, feet (the indexer's geometry)
          -> SigLIP 2 image embeddings, the same space as the index
then search.service treats each crop as one condition and scores every tracked person on all of them,
the way it scores "a red t-shirt" or "a Nike logo".

This matches appearance: clothes, shoes, bags, hair, build. It works when the photo is from the same
day; it does not recognize faces.
"""
from __future__ import annotations

import io
import json
import secrets
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from indexer.geometry import part_boxes

ROOT = Path.home() / "data/queries"
POSE = "yolo26s-pose.pt"
MAX_SIDE = 1600
PARTS = ("head", "torso", "legs", "feet")   # hands say little about who someone is
MIN_PART_PX = 24
PAD = 0.1


class Photos:
    def __init__(self) -> None:
        ROOT.mkdir(parents=True, exist_ok=True)
        self._pose = None
        self.lock = threading.Lock()

    def pose(self):
        with self.lock:
            if self._pose is None:
                from ultralytics import YOLO
                self._pose = YOLO(str(ROOT / POSE) if (ROOT / POSE).exists() else POSE)
            return self._pose

    def add(self, data: bytes) -> dict:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")   # phone photos carry rotation
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        pid = secrets.token_hex(6)
        img.save(ROOT / f"{pid}.jpg", quality=92)
        res = self.pose().predict(np.asarray(img)[:, :, ::-1], imgsz=960, conf=0.3, verbose=False)[0]
        w, h = img.size
        people = []
        if res.boxes is not None and len(res.boxes):
            for box, conf, kp in zip(res.boxes.xyxy.numpy(), res.boxes.conf.numpy(), res.keypoints.data.numpy()):
                people.append({"box": [float(box[0]) / w, float(box[1]) / h, float(box[2]) / w, float(box[3]) / h],
                               "conf": round(float(conf), 3), "kp": kp.tolist()})
        people.sort(key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
        meta = {"id": pid, "width": w, "height": h, "people": people}
        (ROOT / f"{pid}.json").write_text(json.dumps(meta))
        return {"id": pid, "width": w, "height": h, "people": [{"box": p["box"], "conf": p["conf"]} for p in people]}

    def crops(self, pid: str, person: int = 0) -> dict[str, Image.Image]:
        """The chosen person's crops by scale ("whole" plus the parts the photo shows). With nobody
        detected, the whole photo is taken to be the person."""
        if not pid.isalnum():
            raise ValueError("bad photo id")
        img = Image.open(ROOT / f"{pid}.jpg").convert("RGB")
        meta = json.loads((ROOT / f"{pid}.json").read_text())
        w, h = img.size
        if not meta["people"]:
            return {"whole": img}
        p = meta["people"][min(person, len(meta["people"]) - 1)]
        x1, y1, x2, y2 = p["box"][0] * w, p["box"][1] * h, p["box"][2] * w, p["box"][3] * h
        px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
        out = {"whole": img.crop((max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)))}
        for part, side, (a, b, c, d), _ in part_boxes(np.array(p["kp"], dtype=np.float64), (x1, y1, x2, y2)):
            box = (max(0, int(a)), max(0, int(b)), min(w, int(c)), min(h, int(d)))
            if part in PARTS and part not in out and min(box[2] - box[0], box[3] - box[1]) >= MIN_PART_PX:
                out[part] = img.crop(box)
        return out

    def jpeg(self, pid: str, person: int = 0) -> bytes:
        buf = io.BytesIO()
        self.crops(pid, person)["whole"].save(buf, "JPEG", quality=90)
        return buf.getvalue()
