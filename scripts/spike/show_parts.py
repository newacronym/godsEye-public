"""Contact sheet of the body-part views that best match a phrase (a quick look at what retrieval sees).

    PYTHONPATH=services uv run scripts/spike/show_parts.py hand "a hand with a tattoo" [n]
    -> ~/data/logs/parts_<part>_<phrase>.jpg, each cell captioned rank, camera, view id, side, similarity
"""
import io
import sys
from pathlib import Path

import numpy as np
import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector
from PIL import Image, ImageDraw, ImageFont

from search.encoder import TextEncoder

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
CELL, COLS = 200, 8


def main() -> None:
    part, phrase = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    q = HalfVector(TextEncoder().encode([phrase])[0].astype(np.float16))
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        rows = conn.execute("""select id, camera, track, side, px, -(embedding <#> %s), jpeg from part_crops
                               where part = %s order by embedding <#> %s limit %s""", (q, part, q, n)).fetchall()
    font = ImageFont.load_default(size=13)
    sheet = Image.new("RGB", (COLS * CELL, ((len(rows) + COLS - 1) // COLS) * (CELL + 18)), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, (pid, cam, track, side, px, score, jpeg) in enumerate(rows):
        img = Image.open(io.BytesIO(bytes(jpeg))).convert("RGB")
        k = CELL / max(img.size)
        img = img.resize((round(img.size[0] * k), round(img.size[1] * k)), Image.Resampling.LANCZOS)
        x, y = (i % COLS) * CELL, (i // COLS) * (CELL + 18)
        sheet.paste(img, (x + (CELL - img.size[0]) // 2, y))
        d.text((x + 2, y + CELL + 2), f"{i + 1} {cam} p{pid} t{track} {side or ''} {px}px {score:.3f}",
               fill=(255, 255, 255), font=font)
    out = Path.home() / f"data/logs/parts_{part}_{phrase.replace(' ', '_')[:40]}.jpg"
    sheet.save(out, quality=88)
    print(out)


if __name__ == "__main__":
    main()
