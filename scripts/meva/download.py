#!/usr/bin/env python3
"""Download a MEVA slice (date, time window, site, cameras) from the public S3 bucket.

Stdlib only, so it runs anywhere. Skips files already downloaded and the ~2 MB IR/empty clips,
and writes manifest.csv (camera, site, start, end, path, bytes) next to the clips.

Example:
    python3 scripts/meva/download.py --date 2018-03-11 --start 16:15 --end 16:45 --site school
"""
import argparse
import concurrent.futures as cf
import csv
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

BUCKET = "https://mevadata-public-01.s3.amazonaws.com"
PREFIX = "drops-123-r13"
NAME = re.compile(
    r"(?P<date>\d{4}-\d\d-\d\d)\.(?P<start>\d\d-\d\d-\d\d)\.(?P<end>\d\d-\d\d-\d\d)"
    r"\.(?P<site>\w+)\.(?P<cam>G\d+)\.r13\.avi$"
)
MIN_BYTES = 10_000_000


def list_objects(date: str, hour: int) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": f"{PREFIX}/{date}/{hour:02d}/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        xml = urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(params)}").read().decode()
        out += list(zip(re.findall(r"<Key>(.*?)</Key>", xml), map(int, re.findall(r"<Size>(.*?)</Size>", xml))))
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return out
        token = m.group(1)


def download(key: str, size: int, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(f"{BUCKET}/{key}") as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return "ok"


def list_clips(date: str, start: str, end: str, site: str | None = None,
               cameras: set[str] | None = None) -> dict[str, tuple[str, int, re.Match]]:
    """Clips starting in [start, end) as {file name: (S3 key, bytes, filename match)}."""
    t0 = datetime.strptime(f"{date} {start}", "%Y-%m-%d %H:%M")
    t1 = datetime.strptime(f"{date} {end}", "%Y-%m-%d %H:%M")
    # A clip can sit in the folder of the hour it ends in, so list one hour either side.
    hours = range((t0 - timedelta(hours=1)).hour, t1.hour + 1)
    picked = {}
    for h in hours:
        for key, size in list_objects(date, h):
            m = NAME.search(key)
            if not m or size < MIN_BYTES:
                continue
            clip_start = datetime.strptime(f"{m['date']} {m['start']}", "%Y-%m-%d %H-%M-%S")
            if not (t0 <= clip_start < t1) or (site and m["site"] != site) or (cameras and m["cam"] not in cameras):
                continue
            picked[key.rsplit("/", 1)[1]] = (key, size, m)
    return picked


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--start", required=True, help="HH:MM, clips starting at or after this")
    p.add_argument("--end", required=True, help="HH:MM, clips starting before this")
    p.add_argument("--site", help="school | bus | hospital | admin")
    p.add_argument("--cameras", help="comma-separated, e.g. G330,G639")
    p.add_argument("--out", default=str(Path.home() / "data/meva/raw"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    picked = list_clips(a.date, a.start, a.end, a.site, set(a.cameras.split(",")) if a.cameras else None)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(s for _, s, _ in picked.values())
    print(f"{len(picked)} clips, {total / 1e9:.1f} GB -> {out}", flush=True)

    done = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(download, k, s, out / name): (name, s) for name, (k, s, _) in picked.items()}
        for f in cf.as_completed(futs):
            name, size = futs[f]
            status = f.result()
            done += size
            print(f"[{done / total:6.1%}] {status} {name}", flush=True)

    manifest = out / "manifest.csv"
    new = not manifest.exists() or manifest.stat().st_size == 0
    seen = set()
    if not new:
        with open(manifest, newline="") as f:
            seen = {row["file"] for row in csv.DictReader(f)}
    with open(manifest, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "camera", "site", "start", "end", "bytes"])
        for name, (_, size, m) in sorted(picked.items()):
            if name not in seen:
                w.writerow([name, m["cam"], m["site"], f"{m['date']}T{m['start'].replace('-', ':')}",
                            f"{m['date']}T{m['end'].replace('-', ':')}", size])


if __name__ == "__main__":
    sys.exit(main())
