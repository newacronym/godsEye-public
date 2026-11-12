"""Detail-search evaluation against mined, human-confirmed ground truth.

MEVA doesn't label tattoos or glasses, so ground truth comes from mine_details.py (Haiku screens every
stored view) followed by a person confirming the flagged views. A ground-truth file lists the confirmed
people by their stable per-camera track id:

    ~/data/logs/mining/<tag>_gt.json   {"tracks": [["G420", 1234], ...], "ignore": [...], "queries": ["...", ...]}

People in "ignore" (unsure on inspection) count neither as hits nor as false positives.

For each query this runs the shipped pipeline (search.service.run) and reports, per query:
  rank      where each ground-truth person lands in the retrieved list (the verifier sees the top 20)
  verified  recall and precision of the verified matches; "possible" (verifier unsure) counted separately
  before    the rank the same person gets from whole-person search (the pipeline before body parts)

    PYTHONPATH=services uv run scripts/spike/eval_details.py --tag tattoo
"""
import argparse
import json
from pathlib import Path

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from search import service
from search.encoder import TextEncoder
from search.llm import Claude
from search.retrieve import search

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
MINING = Path.home() / "data/logs/mining"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--verify-top", type=int, default=20)
    a = p.parse_args()

    gt_file = json.loads((MINING / f"{a.tag}_gt.json").read_text())
    pool = ConnectionPool(DSN, min_size=1, max_size=16, configure=register_vector, open=True)
    with pool.connection() as conn:
        ids = {(cam, tid): i for i, cam, tid in conn.execute("select id, camera, track_id from tracks")}
    gt = {ids[tuple(k)] for k in gt_file["tracks"]}
    ignore = {ids[tuple(k)] for k in gt_file.get("ignore", [])}
    enc, haiku, f = TextEncoder(), Claude(), service.footage(pool)
    out = []
    for q in gt_file["queries"]:
        matched, verdicts, plan, part = [], {}, None, None
        for ev in service.run(service.Request(q=q, verify_top=a.verify_top), pool, enc, haiku, f):
            if ev["type"] == "plan":
                plan = ev["plan"]
            elif ev["type"] == "matched":
                matched, part = ev["events"], ev["mode"]
            elif ev["type"] == "verdict":
                verdicts[ev["id"]] = ev
        track_of = lambda e: int(e["id"][1:]) if e["id"].startswith("t") else None
        ranks = {t: next((i + 1 for i, e in enumerate(matched) if track_of(e) == t), None) for t in gt}
        verified = {track_of(e) for e in matched if verdicts.get(e["id"], {}).get("match")} - ignore
        possible = {track_of(e) for e in matched if verdicts.get(e["id"], {}).get("unsure")} - ignore - verified
        hits = verified & gt
        # Before body parts: whole-person crops and tiles, searched with the plan's object phrase.
        with pool.connection() as conn:
            before = search(conn, enc.encode([(plan or {}).get("object_phrase") or q])[0], limit=200)
        before_ranks = {t: next((i + 1 for i, e in enumerate(before) if e.track == t), None) for t in gt}
        row = {"query": q, "mode": part, "conditions": [(a["phrase"], a["scale"]) for a in (plan or {}).get("attributes", [])],
               "gt": len(gt), "ranks": sorted(r or 999 for r in ranks.values()),
               "before_ranks": sorted(r or 999 for r in before_ranks.values()),
               "verified": len(verified), "verified_hits": len(hits),
               "recall": round(len(hits) / len(gt), 2) if gt else None,
               "precision": round(len(hits) / len(verified), 2) if verified else None,
               "possible": len(possible), "possible_hits": len(possible & gt),
               "recall_with_possible": round(len((verified | possible) & gt) / len(gt), 2) if gt else None,
               "false_positives": sorted(verified - gt),
               "gt_reasons": {str(t): verdicts.get(f"t{t}", {}).get("reason") for t in gt if ranks[t] and ranks[t] <= a.verify_top}}
        out.append(row)
        print(json.dumps(row))
    (MINING / f"{a.tag}_eval.json").write_text(json.dumps(out, indent=2))
    pool.close()


if __name__ == "__main__":
    main()
