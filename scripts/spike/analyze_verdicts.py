"""Summarize verifier verdicts from an eval run: confusion counts, and sample reasons for misses.

    uv run scripts/spike/analyze_verdicts.py v3
"""
import collections
import json
import sys
from pathlib import Path

tag = sys.argv[1] if len(sys.argv) > 1 else "v3"
rows = [json.loads(line) for line in open(Path.home() / f"data/logs/eval_candidates_{tag}.jsonl")]
tp = [r for r in rows if r["label_hit"] and r["match"]]
fp = [r for r in rows if not r["label_hit"] and r["match"]]
fn = [r for r in rows if r["label_hit"] and not r["match"]]
print(f"{len(rows)} verdicts: TP {len(tp)}  FP {len(fp)}  FN {len(fn)}  "
      f"precision {len(tp) / max(len(tp) + len(fp), 1):.2f}  recall {len(tp) / max(len(tp) + len(fn), 1):.2f}")

by_intent = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    c = by_intent[r["intent"]]
    c[0] += r["label_hit"] and bool(r["match"])
    c[1] += (not r["label_hit"]) and bool(r["match"])
    c[2] += r["label_hit"] and not r["match"]
for intent, (a, b, c) in by_intent.items():
    print(f"  {intent:10} TP {a:3} FP {b:3} FN {c:3}")


def show(title: str, rs: list, n: int) -> None:
    print(f"--- {title}")
    for r in rs[:n]:
        print(f"  [{r['query'][:34]:34}] rank {r['rank']:>2} {r['kind']:6} {r['camera']} {r['moment'][11:19]} "
              f"conf {r['confidence']}: {r['reason']}")


show("labeled correct but rejected (FN)", fn, 12)
show("accepted but not labeled (FP)", fp, 8)
