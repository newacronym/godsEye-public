#!/usr/bin/env python3
"""Download and flatten MEVA (Kitware, KPF YAML) annotations for our downloaded clips.

Outputs, in ~/data/meva/annotations/:
  events.csv     one row per activity instance: clip, camera, activity, frames, wall-clock times, actors
  boxes.csv.gz   actor boxes (every 5th frame) for actors that take part in an activity; used to check
                 that a search result points at the right person/vehicle

Stdlib only; the YAML is one record per line, so regexes are enough and much faster than a YAML parser.

Example:
    python3 scripts/meva/annotations.py --date 2018-03-11 --hour 16
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import gzip
import re
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

REPO = "https://gitlab.kitware.com/meva/meva-data-repo/-/raw/master/annotation/DIVA-phase-2/MEVA/kitware"
FPS = 30
BOX_STRIDE = 5

ACT = re.compile(r"act2: \{(?P<name>\w+): [\d.]+\}, id2: (?P<id>\d+), timespan: \[\{tsr0: \[(?P<s>\d+), (?P<e>\d+)\]\}\]")
ACTOR = re.compile(r"\{id1: (\d+), timespan")
TYPE = re.compile(r"types: \{ id1: (\d+), cset3: \{ (\w+):")
GEOM = re.compile(r"id1: (\d+), id0: \d+, ts0: (\d+),.*?g0: (\d+) (\d+) (\d+) (\d+)")


def fetch(url: str, dest: Path) -> None:
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)


def clip_start(clip: str) -> datetime:
    date, start = clip.split(".")[:2]
    return datetime.strptime(f"{date} {start}", "%Y-%m-%d %H-%M-%S")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--hour", required=True, type=int)
    p.add_argument("--clips", default=str(Path.home() / "data/meva/annotated_clips.txt"))
    p.add_argument("--out", default=str(Path.home() / "data/meva/annotations"))
    a = p.parse_args()

    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    clips = [c for c in Path(a.clips).read_text().split() if c]

    jobs = [(f"{REPO}/{a.date}/{a.hour:02d}/{c}.{kind}.yml", out / "raw" / f"{c}.{kind}.yml")
            for c in clips for kind in ("activities", "types", "geom")]
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))

    counts = collections.Counter()
    with open(out / "events.csv", "w", newline="") as ef, gzip.open(out / "boxes.csv.gz", "wt", newline="") as bf:
        ev = csv.writer(ef)
        ev.writerow(["event_id", "clip", "camera", "activity", "start_frame", "end_frame",
                     "start_time", "end_time", "duration_s", "actors"])
        bx = csv.writer(bf)
        bx.writerow(["clip", "actor_id", "frame", "x1", "y1", "x2", "y2"])
        for clip in clips:
            raw = out / "raw"
            types = dict(TYPE.findall((raw / f"{clip}.types.yml").read_text()))
            t0 = clip_start(clip)
            in_activity = set()
            for line in (raw / f"{clip}.activities.yml").read_text().splitlines():
                m = ACT.search(line)
                if not m:
                    continue
                s, e = int(m["s"]), int(m["e"])
                actors = ACTOR.findall(line)
                in_activity.update(actors)
                counts[m["name"]] += 1
                ev.writerow([m["id"], clip, clip.split(".")[-1], m["name"], s, e,
                             (t0 + timedelta(seconds=s / FPS)).isoformat(),
                             (t0 + timedelta(seconds=e / FPS)).isoformat(),
                             round((e - s) / FPS, 2), ";".join(f"{x}:{types.get(x, '?')}" for x in actors)])
            for line in (raw / f"{clip}.geom.yml").read_text().splitlines():
                m = GEOM.search(line)
                if m and m[1] in in_activity and int(m[2]) % BOX_STRIDE == 0:
                    bx.writerow([clip, *m.groups()])

    print(f"{sum(counts.values())} activity instances in {len(clips)} clips -> {out}/events.csv")
    for name, n in counts.most_common():
        print(f"  {n:4d}  {name}")


if __name__ == "__main__":
    main()
