"""Load Modal indexer output (Parquet per camera) into Postgres on the Studio.

    modal volume get godseye-index spike ~/data/index/      # fetch first (parts.parquet is optional)
    uv run services/indexer/load.py --run spike             # then load (idempotent per camera)
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
import pyarrow.parquet as pq
from pgvector import HalfVector
from pgvector.psycopg import register_vector

DSN = "host=127.0.0.1 port=5432 user=postgres dbname=godseye"
SCHEMA = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
HNSW = {
    "track_crops_hnsw": "create index if not exists track_crops_hnsw on track_crops using hnsw (embedding halfvec_ip_ops)",
    "frame_embeds_hnsw": "create index if not exists frame_embeds_hnsw on frame_embeds using hnsw (embedding halfvec_ip_ops)",
}


def ts(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def load_camera(conn: psycopg.Connection, run: str, cam_dir: Path) -> dict:
    camera = cam_dir.name
    tracks = pq.read_table(cam_dir / "tracks.parquet").to_pylist()
    crops = pq.read_table(cam_dir / "crops.parquet").to_pylist()
    frames = pq.read_table(cam_dir / "frames.parquet").to_pylist()
    with conn.cursor() as cur:
        cur.execute("delete from frame_embeds where run = %s and camera = %s", (run, camera))
        cur.execute("delete from tracks where run = %s and camera = %s", (run, camera))  # cascades to crops
        with cur.copy("copy tracks (run, camera, track_id, class, conf_mean, start_ts, end_ts, traj_t0,"
                      " traj_dt_ms, traj_xywh, best_jpeg) from stdin") as cp:
            for t in tracks:
                cp.write_row((run, camera, t["track_id"], t["class"], t["conf_mean"], ts(t["start_ms"]),
                              ts(t["end_ms"]), ts(t["start_ms"]), t["traj_dt_ms"], t["traj_xywh"], t["best_jpeg"]))
        cur.execute("select track_id, id from tracks where run = %s and camera = %s", (run, camera))
        ids = dict(cur.fetchall())
        with cur.copy("copy track_crops (track, run, camera, ts, bbox, jpeg, embedding) from stdin") as cp:
            for c in crops:
                cp.write_row((ids[c["track_id"]], run, camera, ts(c["ts_ms"]), c["bbox"], c["jpeg"],
                              HalfVector(np.asarray(c["embedding"], dtype=np.float16))))
        with cur.copy("copy frame_embeds (run, camera, ts, tile, box, motion, thumb, embedding) from stdin") as cp:
            for f in frames:
                cp.write_row((run, camera, ts(f["ts_ms"]), f["tile"], f["box"], f["motion"], f["thumb"],
                              HalfVector(np.asarray(f["embedding"], dtype=np.float16))))
        parts = pq.read_table(cam_dir / "parts.parquet").to_pylist() if (cam_dir / "parts.parquet").exists() else []
        with cur.copy("copy part_crops (track, run, camera, part, side, ts, box, px, quality, jpeg, embedding)"
                      " from stdin") as cp:
            for r in parts:
                cp.write_row((ids[r["track_id"]], run, camera, r["part"], r["side"], ts(r["ts_ms"]), r["box"],
                              r["px"], r["quality"], r["jpeg"], HalfVector(np.asarray(r["embedding"], dtype=np.float16))))
    conn.commit()
    return {"camera": camera, "tracks": len(tracks), "crops": len(crops), "frames": len(frames), "parts": len(parts)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="spike")
    p.add_argument("--root", default=str(Path.home() / "data/index"))
    a = p.parse_args()

    with psycopg.connect(DSN) as conn:
        conn.execute(SCHEMA.read_text())
        conn.commit()
        register_vector(conn)
        cams = sorted(d for d in (Path(a.root) / a.run).iterdir() if (d / "tracks.parquet").exists())
        for cam_dir in cams:
            t = time.perf_counter()
            stats = load_camera(conn, a.run, cam_dir)
            print(json.dumps(stats | {"load_s": round(time.perf_counter() - t, 1)}))
        t = time.perf_counter()
        for sql in HNSW.values():
            conn.execute(sql)
        conn.commit()
        print(f"HNSW indexes ready in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
