"""Rebuild the verifier's evidence for one eval candidate as a single contact sheet, and list every MEVA
label on that camera around that moment, to judge whether the verifier or the labels are wrong.

    PYTHONPATH=services uv run scripts/spike/show_evidence.py "a person looking down at their phone" 4
"""
import csv
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from PIL import Image, ImageDraw
from pgvector.psycopg import register_vector

from search.encoder import TextEncoder
from search.evidence import gather
from search.retrieve import search

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
OUT = Path.home() / "data/logs/evidence"


def labeled(ev, covered) -> bool:
    return any(s <= ev.moment < e for s, e in covered.get(ev.camera, []))


def main() -> None:
    query, rank = sys.argv[1], int(sys.argv[2])
    intent = sys.argv[3] if len(sys.argv) > 3 else "activity"
    covered = {}
    for c in (Path.home() / "data/meva/annotated_clips.txt").read_text().split():
        date, start = c.split(".")[:2]
        t0 = datetime.strptime(f"{date} {start}", "%Y-%m-%d %H-%M-%S").replace(tzinfo=timezone.utc)
        covered.setdefault(c.split(".")[-1], []).append((t0, t0 + timedelta(minutes=5)))

    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        ranked = [e for e in search(conn, TextEncoder().encode([query])[0], limit=200) if labeled(e, covered)]
        ev = ranked[rank]
        images = gather(conn, ev, intent)

    print(f"{query!r} rank {rank}: {ev.kind} {ev.camera} moment {ev.moment:%H:%M:%S} "
          f"interval {ev.start:%H:%M:%S}-{ev.end:%H:%M:%S} score {ev.score:.3f} cls {ev.cls}")
    print("MEVA labels on this camera within 5 s of the moment:")
    for r in csv.DictReader(open(Path.home() / "data/meva/annotations/events.csv")):
        s = datetime.fromisoformat(r["start_time"]).replace(tzinfo=timezone.utc)
        e = datetime.fromisoformat(r["end_time"]).replace(tzinfo=timezone.utc)
        if r["camera"] == ev.camera and s - timedelta(seconds=5) <= ev.moment <= e + timedelta(seconds=5):
            print(f"  {r['activity']:40} {s:%H:%M:%S}-{e:%H:%M:%S}  actors {r['actors'][:60]}")

    tiles = [Image.open(io.BytesIO(j)).convert("RGB") for _, j in images]
    cell = 300
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 30) + 30), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), f"{query}  | rank {rank} {ev.kind} {ev.camera} {ev.moment:%H:%M:%S}", fill=(255, 200, 0))
    for i, ((caption, _), t) in enumerate(zip(images, tiles)):
        t.thumbnail((cell, cell))
        x, y = (i % cols) * cell, 30 + (i // cols) * (cell + 30)
        sheet.paste(t, (x, y))
        d.text((x + 4, y + cell + 4), caption[:48], fill=(220, 220, 220))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{query[:30].replace(' ', '_')}_r{rank}.jpg"
    sheet.save(path, quality=85)
    print(path)


if __name__ == "__main__":
    main()
