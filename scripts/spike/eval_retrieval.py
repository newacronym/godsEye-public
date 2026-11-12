"""Spike evaluation of search against MEVA labels.

A prediction hits a labeled instance when: same camera, same activity, its representative moment lies
inside the instance interval +/- 2 s, and (strict mode, track results only) its box overlaps one of the
instance's actors (IoU >= 0.3 at the nearest labeled frame). Matching is one-to-one in rank order.
Predictions from clips without labels are dropped before scoring.

    PYTHONPATH=services uv run scripts/spike/eval_retrieval.py                    # retrieval only
    PYTHONPATH=services uv run scripts/spike/eval_retrieval.py --verify --tag v3  # + planner and verifier

With --verify, the planner picks the evidence format (appearance, state or action), the verifier judges
the top --verify-top candidates, and the result list becomes the verified matches, most confident first. Every
verdict is logged next to its label (~/data/logs/eval_candidates_<tag>.jsonl), which gives the
verifier's own precision and recall.
"""
import argparse
import collections
import csv
import gzip
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from search.encoder import TextEncoder
from search.evidence import gather
from search.llm import Claude
from search.retrieve import search

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
ANN = Path.home() / "data/meva/annotations"
RAW = Path.home() / "data/meva/raw"
LOGS = Path.home() / "data/logs"
FPS = 30
TOL = timedelta(seconds=2)
FOOTAGE = "MEVA school site, 2018-03-11 16:15-16:45"

# Dev queries: plain phrasings of the most frequent spike-slice activities (>= 16 instances each).
# Labels are events, so state-like wordings are phrased as the act ("in the middle of sitting down").
QUERIES = {
    "person_carries_heavy_object": ["a person carrying a heavy box", "someone carrying something heavy"],
    "person_texts_on_phone": ["a person looking down at their phone", "someone texting on a phone"],
    "person_sits_down": ["a person in the middle of sitting down onto a chair"],
    "person_talks_to_person": ["two people talking to each other"],
    "person_picks_up_object": ["a person picking something up from the ground"],
    "person_puts_down_object": ["a person putting an object down"],
    "person_opens_facility_door": ["a person opening the door of a building"],
    "person_enters_scene_through_structure": ["a person walking out of a building door"],
    "person_transfers_object": ["a person handing something to another person"],
    "person_opens_vehicle_door": ["a person opening a car door"],
    "vehicle_stops": ["a car stopping"],
    "vehicle_turns_left": ["a car turning left"],
    "person_embraces_person": ["two people hugging"],
    "person_enters_vehicle": ["a person getting into a car"],
    "person_exits_vehicle": ["a person getting out of a car"],
    "person_talks_on_phone": ["a person talking on a phone held to their ear"],
    "person_reads_document": ["a person reading a document"],
    "hand_interacts_with_person": ["two people shaking hands"],
}


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def clip_start(clip: str) -> datetime:
    date, start = clip.split(".")[:2]
    return datetime.strptime(f"{date} {start}", "%Y-%m-%d %H-%M-%S").replace(tzinfo=timezone.utc)


def camera_sizes() -> dict[str, tuple[int, int]]:
    sizes = {}
    for f in sorted(RAW.glob("*.avi")):
        cam = f.name.split(".")[-3]
        if cam not in sizes:
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                  "stream=width,height", "-of", "csv=p=0", str(f)], capture_output=True, text=True)
            w, h = map(int, out.stdout.strip().split(","))
            sizes[cam] = (w, h)
    return sizes


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Labels:
    def __init__(self) -> None:
        annotated = [c for c in (Path.home() / "data/meva/annotated_clips.txt").read_text().split() if c]
        self.covered = collections.defaultdict(list)  # camera -> [(start, end)] of labeled clips
        for c in annotated:
            self.covered[c.split(".")[-1]].append((clip_start(c), clip_start(c) + timedelta(minutes=5)))
        self.events = collections.defaultdict(list)
        for r in csv.DictReader(open(ANN / "events.csv")):
            r["start"], r["end"] = utc(r["start_time"]), utc(r["end_time"])
            r["actor_ids"] = {x.split(":")[0] for x in r["actors"].split(";") if x}
            self.events[r["activity"]].append(r)
        sizes = camera_sizes()
        self.boxes = collections.defaultdict(list)
        with gzip.open(ANN / "boxes.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                w, h = sizes[r["clip"].split(".")[-1]]
                self.boxes[(r["clip"], r["actor_id"])].append(
                    (int(r["frame"]), (int(r["x1"]) / w, int(r["y1"]) / h, int(r["x2"]) / w, int(r["y2"]) / h)))

    def labeled(self, ev) -> bool:
        return any(s <= ev.moment < e for s, e in self.covered[ev.camera])

    def _actor_hit(self, inst, ev) -> bool:
        frame = round((ev.moment - clip_start(inst["clip"])).total_seconds() * FPS)
        for actor in inst["actor_ids"]:
            pts = self.boxes.get((inst["clip"], actor))
            if pts:
                _, box = min(pts, key=lambda p: abs(p[0] - frame))
                if iou(box, ev.bbox) >= 0.3:
                    return True
        return False

    def score(self, activity: str, ranked: list, strict: bool = True) -> list[bool]:
        """One-to-one matching in rank order; True where a prediction claims an unused instance."""
        used, hits = set(), []
        for ev in ranked:
            match = None
            for i, inst in enumerate(self.events[activity]):
                if i in used or inst["camera"] != ev.camera:
                    continue
                if not (inst["start"] - TOL <= ev.moment <= inst["end"] + TOL):
                    continue
                if strict and ev.kind == "track" and not self._actor_hit(inst, ev):
                    continue
                match = i
                break
            if match is not None:
                used.add(match)
            hits.append(match is not None)
        return hits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify-top", type=int, default=30)
    p.add_argument("--tag", default="dev")
    a = p.parse_args()

    labels = Labels()
    enc = TextEncoder()
    haiku = Claude() if a.verify else None
    LOGS.mkdir(parents=True, exist_ok=True)
    cand_log = open(LOGS / f"eval_candidates_{a.tag}.jsonl", "w") if a.verify else None
    rows = []
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        cameras = [r[0] for r in conn.execute("select distinct camera from tracks order by 1")]
        for activity, phrasings in QUERIES.items():
            for q in phrasings:
                t = time.perf_counter()
                vec = enc.encode([q])[0]
                t_enc = time.perf_counter() - t
                t = time.perf_counter()
                ranked = [e for e in search(conn, vec, limit=200) if labels.labeled(e)]
                t_search = time.perf_counter() - t
                strict = labels.score(activity, ranked[:a.k])
                relaxed = labels.score(activity, ranked[:a.k], strict=False)
                row = {"activity": activity, "query": q, "n_gt": len(labels.events[activity]),
                       "P@5": sum(strict[:5]) / 5, "P@10": sum(strict[:10]) / 10,
                       "hit@1": strict[:1] == [True], "hit@5": any(strict[:5]),
                       "P@10_relaxed": sum(relaxed[:10]) / 10,
                       "enc_ms": round(t_enc * 1000), "search_ms": round(t_search * 1000)}
                if haiku:
                    t = time.perf_counter()
                    plan = haiku.plan(q, FOOTAGE, cameras)
                    t_plan = time.perf_counter() - t
                    intent = plan.intent if plan and plan.intent != "count" else "action"
                    cands = ranked[:a.verify_top]
                    t = time.perf_counter()
                    with ThreadPoolExecutor(4) as ex:  # ffmpeg strips: CPU-bound, one per core
                        evid = list(ex.map(lambda e: gather(psycopg.connect(DSN), e, intent), cands))
                    t_evid = time.perf_counter() - t
                    t = time.perf_counter()
                    with ThreadPoolExecutor(10) as ex:
                        conditions = (plan.conditions() if plan else []) or [q]
                        verdicts = list(ex.map(lambda im: haiku.verify(q, conditions, im, intent), evid))
                    t_verify = time.perf_counter() - t
                    truth = labels.score(activity, cands)
                    for rank, (e, v, h, im) in enumerate(zip(cands, verdicts, truth, evid)):
                        cand_log.write(json.dumps({
                            "activity": activity, "query": q, "intent": intent, "rank": rank, "kind": e.kind,
                            "camera": e.camera, "moment": e.moment.isoformat(), "label_hit": h,
                            "match": v.match if v else None, "confidence": v.confidence if v else None,
                            "reason": v.reason if v else None, "n_images": len(im)}) + "\n")
                    # Verified matches, most confident first (ties keep retrieval order).
                    kept = [e for e, v in sorted(zip(cands, verdicts), key=lambda p: -(p[1].confidence if p[1] else 0))
                            if v and v.match]
                    vs = labels.score(activity, kept[:a.k])
                    tp = sum(bool(v and v.match) and h for v, h in zip(verdicts, truth))
                    fp = sum(bool(v and v.match) and not h for v, h in zip(verdicts, truth))
                    fn = sum(not (v and v.match) and h for v, h in zip(verdicts, truth))
                    row |= {"intent": intent, "verified_n": len(kept), "vP@5": sum(vs[:5]) / 5,
                            "vP@10": sum(vs[:10]) / 10, "v_tp": tp, "v_fp": fp, "v_fn": fn,
                            "plan_ms": round(t_plan * 1000), "evidence_ms": round(t_evid * 1000),
                            "verify_ms": round(t_verify * 1000), "failed_calls": sum(v is None for v in verdicts)}
                rows.append(row)
                print(json.dumps(row), flush=True)

    print("\n=== summary (strict = right camera + moment + actor box; relaxed = no box check) ===")
    keys = ["P@5", "P@10", "P@10_relaxed"] + (["vP@5", "vP@10"] if a.verify else [])
    for k in keys:
        print(f"{k:>16}: {statistics.mean(r[k] for r in rows):.2f}")
    for k in ["hit@1", "hit@5"]:
        print(f"{k:>16}: {sum(r[k] for r in rows)}/{len(rows)} queries")
    if a.verify:
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("v_tp", "v_fp", "v_fn"))
        print(f"{'verifier prec.':>16}: {tp / max(tp + fp, 1):.2f}  ({tp} of {tp + fp} accepted were labeled hits)")
        print(f"{'verifier recall':>16}: {tp / max(tp + fn, 1):.2f}  ({tp} of {tp + fn} labeled hits accepted)")
        for k in ("plan_ms", "evidence_ms", "verify_ms"):
            print(f"{k:>16}: median {statistics.median(r[k] for r in rows)}")
    print(f"{'encode ms':>16}: median {statistics.median(r['enc_ms'] for r in rows)} (CPU)")
    print(f"{'search ms':>16}: median {statistics.median(r['search_ms'] for r in rows)}")


if __name__ == "__main__":
    main()
