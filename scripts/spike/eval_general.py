"""General search evaluation: many kinds of question, judged by eye.

MEVA labels activities, not appearance, so for "a man in a grey hoodie and white sneakers" there is no
ground truth to score against. This runs a broad set of questions through the shipped pipeline
(search.service.run), saves a contact sheet of what each one verified and the top possible matches, and
scores them against judgments a person made by looking at those sheets:

    ~/data/logs/general_eval/judgments.json   {"<query>": {"G420#223": true, "G330#17": false, ...}}

Keys are camera#track_id (stable across database rebuilds). Results nobody has judged yet are listed so
they can be added. Reported per query and overall: verified precision, how many were verified and possible,
precision of the top 5 possible, and time to the last verdict.

    PYTHONPATH=services uv run scripts/spike/eval_general.py                 # all queries
    PYTHONPATH=services uv run scripts/spike/eval_general.py --only glasses   # queries containing a word
"""
import argparse
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pgvector.psycopg import register_vector
from PIL import Image, ImageDraw, ImageFont
from psycopg_pool import ConnectionPool

from search import service
from search.encoder import TextEncoder
from search.llm import Claude

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
OUT = Path.home() / "data/logs/general_eval"

QUERIES = {
    "clothing": ["a person in a red top", "someone in a white hoodie", "a man wearing a denim jacket",
                 "a person in a long black coat", "someone wearing camouflage pants"],
    "head": ["a man wearing glasses", "a person in a baseball cap", "a man with a beard",
             "a person wearing a beanie", "a woman with a ponytail"],
    "torso": ["a hoodie with PUMA written on it", "a shirt with a big printed graphic", "a person wearing a lanyard"],
    "hands": ["find a person with a tattoo on his right hand", "someone wearing a wristwatch",
              "a person holding a coffee cup"],
    "feet": ["someone wearing white sneakers", "a person in brown boots"],
    "objects": ["a person with a backpack", "a red backpack", "a person pulling a suitcase"],
    "vehicles": ["a red pickup truck", "a white car", "a black SUV"],
    "combined": ["a man in a red Nike t-shirt", "a man in a grey hoodie and white sneakers",
                 "a woman in a red top carrying a black handbag", "a bald man wearing glasses"],
}
CELL, POSSIBLE_SHOWN = 200, 6


def key(conn, track: int) -> str:
    cam, tid = conn.execute("select camera, track_id from tracks where id = %s", (track,)).fetchone()
    return f"{cam}#{tid}"


def run_one(q: str, pool: ConnectionPool, enc: TextEncoder, claude: Claude, f: service.Footage) -> dict:
    matched, verdicts, plan, mode, done = [], {}, None, None, {}
    for ev in service.run(service.Request(q=q), pool, enc, claude, f):
        if ev["type"] == "plan":
            plan = ev["plan"]
        elif ev["type"] == "matched":
            matched, mode = ev["events"], ev["mode"]
        elif ev["type"] == "verdict":
            verdicts[ev["id"]] = ev
        elif ev["type"] == "done":
            done = ev
    rows = []
    with pool.connection() as conn:
        for rank, e in enumerate(matched):
            v = verdicts.get(e["id"])
            if not v or not (v["match"] or v["unsure"]):
                continue
            k = key(conn, int(e["id"][1:])) if e["id"].startswith("t") else f"{e['camera']}@{e['moment'][11:19]}"
            rows.append({"key": k, "rank": rank + 1, "status": "verified" if v["match"] else "possible",
                         "image": e["image"], "reason": v["reason"], "checks": v.get("checks", []),
                         "confidence": v["confidence"]})
    support = lambda r: sum({"yes": 1, "maybe": 0.5}.get(c["answer"], 0) for c in r["checks"]) / max(1, len(r["checks"]))
    rows.sort(key=lambda r: (r["status"] != "verified", -support(r), -(r["confidence"] or 0)))
    return {"query": q, "mode": mode, "intent": done.get("intent"), "ms": done.get("ms"),
            "conditions": [(a["phrase"], a["scale"]) for a in (plan or {}).get("attributes", [])]
                          + ([(plan["activity"], "activity")] if plan and plan.get("activity") else []),
            "rows": rows}


def sheet(res: dict, pool: ConnectionPool, path: Path) -> None:
    shown = [r for r in res["rows"] if r["status"] == "verified"] + \
            [r for r in res["rows"] if r["status"] == "possible"][:POSSIBLE_SHOWN]
    font = ImageFont.load_default(size=12)
    cols = 8
    img = Image.new("RGB", (cols * CELL, max(1, (len(shown) + cols - 1) // cols) * (CELL + 30)), (30, 30, 30))
    d = ImageDraw.Draw(img)
    with pool.connection() as conn:
        for i, r in enumerate(shown):
            kind, ident = re.match(r"/media/(\w+)/(\d+)\.jpg", r["image"]).groups()
            if kind == "tile":   # a scene window: the nearest whole-frame thumbnail
                jpeg = conn.execute("""select t.thumb from frame_embeds f join lateral (select thumb from frame_embeds t
                    where t.camera = f.camera and t.tile = 0 and t.thumb is not null
                    order by abs(extract(epoch from t.ts - f.ts)) limit 1) t on true where f.id = %s""",
                                    (int(ident),)).fetchone()[0]
            else:
                table = {"crop": "track_crops", "part": "part_crops"}[kind]
                jpeg = conn.execute(f"select jpeg from {table} where id = %s", (int(ident),)).fetchone()[0]
            x, y = (i % cols) * CELL, (i // cols) * (CELL + 30)
            if jpeg:
                im = Image.open(io.BytesIO(bytes(jpeg))).convert("RGB")
                k = CELL / max(im.size)
                im = im.resize((max(1, round(im.size[0] * k)), max(1, round(im.size[1] * k))), Image.Resampling.LANCZOS)
                img.paste(im, (x + (CELL - im.size[0]) // 2, y))
            tag = "V" if r["status"] == "verified" else "?"
            d.text((x + 2, y + CELL + 2), f"{tag}{i + 1} {r['key']}", fill=(255, 255, 0), font=font)
            d.text((x + 2, y + CELL + 15), " ".join(c["answer"][:3] for c in r["checks"]), fill=(200, 200, 200), font=font)
    img.save(path, quality=88)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--score-only", action="store_true", help="rescore the last results.json with current judgments")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    judged = json.loads((OUT / "judgments.json").read_text()) if (OUT / "judgments.json").exists() else {}

    pool = ConnectionPool(DSN, min_size=1, max_size=24, configure=register_vector, open=True)
    if a.score_only:
        results = json.loads((OUT / "results.json").read_text())
    else:
        enc, claude, f = TextEncoder(), Claude(), service.footage(pool)
        queries = [(cat, q) for cat, qs in QUERIES.items() for q in qs if a.only.lower() in q.lower()]
        with ThreadPoolExecutor(a.workers) as ex:
            results = list(ex.map(lambda cq: run_one(cq[1], pool, enc, claude, f) | {"category": cq[0]}, queries))

    totals = {"verified": 0, "v_judged": 0, "v_right": 0, "p_judged": 0, "p_right": 0}
    by_cat: dict[str, list[int]] = {}
    for res in results:
        slug = re.sub(r"[^a-z0-9]+", "_", res["query"].lower()).strip("_")[:40]
        if not a.score_only:
            sheet(res, pool, OUT / f"{slug}.jpg")
        j = judged.get(res["query"], {})
        ver = [r for r in res["rows"] if r["status"] == "verified"]
        pos = [r for r in res["rows"] if r["status"] == "possible"][:5]
        vj = [j[r["key"]] for r in ver if r["key"] in j]
        pj = [j[r["key"]] for r in pos if r["key"] in j]
        res |= {"verified": len(ver), "possible": len([r for r in res["rows"] if r["status"] == "possible"]),
                "v_precision": round(sum(vj) / len(vj), 2) if vj else None,
                "p5_precision": round(sum(pj) / len(pj), 2) if pj else None,
                "unjudged": [r["key"] for r in ver + pos if r["key"] not in j], "sheet": f"{slug}.jpg"}
        c = by_cat.setdefault(res["category"], [0, 0])
        c[0] += len(vj)
        c[1] += sum(vj)
        totals["verified"] += len(ver)
        totals["v_judged"] += len(vj)
        totals["v_right"] += sum(vj)
        totals["p_judged"] += len(pj)
        totals["p_right"] += sum(pj)
        print(f"{res['category']:9s} {res['query'][:44]:44s} {res['mode'] or '-':10s} verified {res['verified']:2d} "
              f"(P {res['v_precision']}) possible {res['possible']:2d} (P@5 {res['p5_precision']}) "
              f"{(res['ms'] or 0) / 1000:5.1f}s  unjudged {len(res['unjudged'])}", flush=True)
    if not a.score_only:
        (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nverified precision by category: " + ", ".join(f"{k} {ok}/{n}" for k, (n, ok) in by_cat.items()))
    if totals["v_judged"]:
        print(f"\nverified precision {totals['v_right'] / totals['v_judged']:.2f} "
              f"({totals['v_right']}/{totals['v_judged']} judged of {totals['verified']}); "
              f"possible top-5 precision {totals['p_right'] / max(1, totals['p_judged']):.2f} ({totals['p_right']}/{totals['p_judged']})")
    pool.close()


if __name__ == "__main__":
    main()
