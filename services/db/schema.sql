-- Index tables. Embeddings are L2-normalized SigLIP 2 vectors stored as halfvec,
-- so inner product equals cosine similarity. HNSW indexes are built by the loader after bulk loads.
create extension if not exists vector;

create table if not exists tracks (
    id          bigserial primary key,
    run         text        not null,
    camera      text        not null,
    track_id    integer     not null,
    class       text        not null,
    conf_mean   real,
    start_ts    timestamptz not null,
    end_ts      timestamptz not null,
    traj_t0     timestamptz not null,   -- trajectory point i is at traj_t0 + traj_dt_ms[i]
    traj_dt_ms  integer[]   not null,
    traj_xywh   real[]      not null,   -- normalized cx, cy, w, h per point, flattened
    best_jpeg   bytea,
    unique (run, camera, track_id)
);
create index if not exists tracks_cam_time on tracks (camera, start_ts, end_ts);

create table if not exists track_crops (
    id        bigserial primary key,
    track     bigint      not null references tracks (id) on delete cascade,
    run       text        not null,
    camera    text        not null,
    ts        timestamptz not null,
    bbox      real[]      not null,     -- normalized x1, y1, x2, y2
    jpeg      bytea,
    embedding halfvec(1152) not null
);
create index if not exists track_crops_cam_time on track_crops (camera, ts);

create table if not exists frame_embeds (
    id        bigserial primary key,
    run       text        not null,
    camera    text        not null,
    ts        timestamptz not null,
    tile      smallint    not null,     -- 0 = full frame, 1-4 = overlapping quadrants
    box       real[]      not null,
    motion    real,
    thumb     bytea,                    -- full-frame thumbnail, tile 0 only
    embedding halfvec(1152) not null
);
create index if not exists frame_embeds_cam_time on frame_embeds (camera, ts);

-- Body-part views: native-resolution crops of hands, heads and torsos of tracked people,
-- for details too small to show in a whole-person crop. Small enough for exact (sequential) kNN,
-- which keeps recall exact; add a partial HNSW index per part if it grows past ~100k rows.
create table if not exists part_crops (
    id        bigserial primary key,
    track     bigint      not null references tracks (id) on delete cascade,
    run       text        not null,
    camera    text        not null,
    part      text        not null,     -- hand | head | torso
    side      char(1),                  -- L | R, the person's own (pose keypoints); hands only
    ts        timestamptz not null,
    box       real[]      not null,     -- normalized x1, y1, x2, y2 in the frame
    px        smallint    not null,     -- native pixels on the crop's short side
    quality   real,
    jpeg      bytea,
    embedding halfvec(1152) not null
);
create index if not exists part_crops_part on part_crops (part, camera, ts);
create index if not exists part_crops_track on part_crops (track);
