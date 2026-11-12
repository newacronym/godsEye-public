"""Mine ground truth for a body-part detail (e.g. tattoos on hands) that MEVA doesn't label.

Every stored view of the part is shown to Haiku in numbered grids, with an inclusive instruction
(recall over precision); the flagged views are written out with a contact sheet so a person can
confirm them. The confirmed set is what eval_details.py measures recall against.

    PYTHONPATH=services uv run scripts/spike/mine_details.py --part hand --detail "a tattoo" --tag tattoo
    PYTHONPATH=services uv run scripts/spike/mine_details.py --part hand --detail "a tattoo" --tag tattoo_strict \
        --within tattoo --mode strict --model claude-sonnet-5-5 --min-px 56     # second, stricter pass
"""
import argparse
import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import psycopg
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from search.llm import FAST, MODEL, api_key

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
OUT = Path.home() / "data/logs/mining"
CELL, COLS, PER_GRID = 192, 4, 16


class Flags(BaseModel):
    cells: list[int]


SYSTEM = {
    "inclusive": """You screen numbered close-ups cut from CCTV footage for one visual detail. Each cell is one
crop of a person's {part}, numbered in its top-left corner. List every cell number where {detail} is
visible or plausibly visible on the {part}. Recall matters more than precision: a person will double-check
every cell you list. List nothing if no cell shows it.""",
    "strict": """You label numbered close-ups cut from CCTV footage for one visual detail. Each cell is one crop
of a person's {part}, numbered in its top-left corner. List only the cells where you can actually see {detail}
on the {part}. Do not list cells that only show sleeves, gloves, dark or tanned skin, shadows, dirt, blur,
watches, bracelets or held objects. List nothing if no cell clearly shows it.""",
}


def grid(jpegs: list[bytes]) -> bytes:
    rows = (len(jpegs) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * CELL, rows * CELL), (40, 40, 40))
    font = ImageFont.load_default(size=26)
    d = ImageDraw.Draw(sheet)
    for i, jpeg in enumerate(jpegs):
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        k = CELL / max(img.size)
        img = img.resize((max(1, round(img.size[0] * k)), max(1, round(img.size[1] * k))), Image.Resampling.LANCZOS)
        x, y = (i % COLS) * CELL, (i // COLS) * CELL
        sheet.paste(img, (x + (CELL - img.size[0]) // 2, y + (CELL - img.size[1]) // 2))
        d.rectangle([x, y, x + 34, y + 30], fill=(0, 0, 0))
        d.text((x + 4, y + 1), str(i + 1), fill=(255, 255, 0), font=font)
        d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(90, 90, 90))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="hand")
    p.add_argument("--detail", default="a tattoo (ink on the skin)")
    p.add_argument("--tag", default="tattoo")
    p.add_argument("--min-px", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--mode", choices=list(SYSTEM), default="inclusive")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--within", help="only views flagged by an earlier run with this tag")
    a = p.parse_args()

    within = json.loads((OUT / f"{a.within}_flagged.json").read_text()) if a.within else None
    with psycopg.connect(DSN) as conn:
        rows = conn.execute("""select id, jpeg from part_crops where part = %s and px >= %s
                               and (%s::bigint[] is null or id = any(%s)) order by camera, ts""",
                            (a.part, a.min_px, within, within)).fetchall()
    batches = [rows[i:i + PER_GRID] for i in range(0, len(rows), PER_GRID)]
    print(f"{len(rows)} {a.part} views in {len(batches)} grids")
    client = anthropic.Anthropic(api_key=api_key(), max_retries=8)
    system = SYSTEM[a.mode].format(part=a.part, detail=a.detail)
    # Haiku turns thinking off with "disabled"; newer models with "between_tools" (no thinking before replying).
    settings = FAST if a.model == MODEL else {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}

    def screen(batch: list[tuple[int, bytes]]) -> list[int]:
        img = base64.standard_b64encode(grid([bytes(j) for _, j in batch])).decode()
        r = client.messages.parse(
            model=a.model, max_tokens=256, **settings, system=system, output_format=Flags,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}},
                {"type": "text", "text": f"Cells 1-{len(batch)}. Which show {a.detail}?"}]}])
        cells = r.parsed_output.cells if r.parsed_output else []
        return [batch[c - 1][0] for c in cells if 1 <= c <= len(batch)]

    with ThreadPoolExecutor(a.workers) as ex:
        flagged = sorted({pid for ids in ex.map(screen, batches) for pid in ids})
    print(f"{len(rows)} screened", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}_flagged.json").write_text(json.dumps(flagged))
    print(f"flagged {len(flagged)} views -> {OUT / f'{a.tag}_flagged.json'}")
    with psycopg.connect(DSN) as conn:
        jpegs = dict(conn.execute("select id, jpeg from part_crops where id = any(%s)", (flagged,)).fetchall())
    for i in range(0, len(flagged), PER_GRID):
        ids = flagged[i:i + PER_GRID]
        (OUT / f"{a.tag}_flagged_{i // PER_GRID:02d}.jpg").write_bytes(grid([bytes(jpegs[x]) for x in ids]))
        (OUT / f"{a.tag}_flagged_{i // PER_GRID:02d}.txt").write_text("\n".join(f"{n + 1}: {x}" for n, x in enumerate(ids)))


if __name__ == "__main__":
    main()
