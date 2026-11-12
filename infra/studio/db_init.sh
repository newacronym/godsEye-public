#!/usr/bin/env bash
# Creates the godseye database and pgvector extension if missing.
set -euo pipefail
PSQL=(/usr/lib/postgresql/17/bin/psql -h 127.0.0.1 -U postgres -v ON_ERROR_STOP=1 -qtA)
"${PSQL[@]}" -c "select 1 from pg_database where datname = 'godseye'" | grep -q 1 \
  || "${PSQL[@]}" -c "create database godseye"
"${PSQL[@]}" -d godseye -c "create extension if not exists vector"
echo "[db-init] godseye ready (pgvector $("${PSQL[@]}" -d godseye -c "select extversion from pg_extension where extname = 'vector'"))"

# The index is derived data: rebuild it from Parquet when the database is new or was lost.
# Also when body-part files exist that the database doesn't have yet.
has_tracks=$("${PSQL[@]}" -d godseye -c "select to_regclass('tracks') is not null and exists (select 1 from tracks)" 2>/dev/null || echo f)
has_parts=$("${PSQL[@]}" -d godseye -c "select to_regclass('part_crops') is not null and exists (select 1 from part_crops)" 2>/dev/null || echo f)
parts_on_disk=$(compgen -G "$HOME/data/index/*/*/parts.parquet" >/dev/null && echo t || echo f)
if [[ $has_tracks != t || ( $parts_on_disk == t && $has_parts != t ) ]]; then
  for run in "$HOME"/data/index/*/; do
    [[ -d $run ]] || continue
    echo "[db-init] loading index run $(basename "$run") from Parquet"
    (cd "$HOME/godsEye" && uv run services/indexer/load.py --run "$(basename "$run")")
  done
fi
