"""Contact sheet of exactly what the detail verifier sees for given people (tracks.id).

    PYTHONPATH=services uv run scripts/spike/show_detail_evidence.py hand "a hand with a tattoo" 25262 18032
    -> ~/data/logs/evidence/detail_<track>.jpg
"""
import io
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from PIL import Image, ImageDraw, ImageFont

from search.encoder import TextEncoder
from search.evidence import gather
from search.retrieve import Query, search_attributes

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
OUT = Path.home() / "data/logs/evidence"
H = 300


def main() -> None:
    part, phrase, tracks = sys.argv[1], sys.argv[2], {int(t) for t in sys.argv[3:]}
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=13)
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        q = Query(phrase, part, None, TextEncoder().encode([phrase])[0])
        events = [e for e in search_attributes(conn, [q], "person", limit=2000) if e.track in tracks]
        for ev in events:
            imgs = [(cap, Image.open(io.BytesIO(j)).convert("RGB")) for cap, j in gather(conn, ev, "appearance", [q])]
            imgs = [(cap, im.resize((round(im.size[0] * H / im.size[1]), H))) for cap, im in imgs]
            sheet = Image.new("RGB", (sum(im.size[0] + 8 for _, im in imgs), H + 46), (30, 30, 30))
            d, x = ImageDraw.Draw(sheet), 0
            for cap, im in imgs:
                sheet.paste(im, (x, 46))
                words, lines = cap.split(), [""]
                for w in words:
                    lines[-1:] = [lines[-1] + " " + w] if len(lines[-1]) + len(w) < im.size[0] // 7 else [lines[-1], w]
                d.multiline_text((x + 2, 2), "\n".join(l.strip() for l in lines[:3]), fill=(255, 255, 0), font=font)
                x += im.size[0] + 8
            path = OUT / f"detail_{ev.track}.jpg"
            sheet.save(path, quality=90)
            print(path, ev.camera, round(ev.score, 4))


if __name__ == "__main__":
    main()
