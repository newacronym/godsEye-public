#!/usr/bin/env bash
# Idempotent setup for the Lightning Studio (Ubuntu 24.04). Safe to re-run; a no-op once installed.
set -euo pipefail

MEDIAMTX_VERSION=v1.21.1
PROCESS_COMPOSE_VERSION=v1.122.0
PG_MAJOR=17

DATA="$HOME/data"     # persistent (Lightning's synced home): media, index Parquet, secrets, logs
OPT="$HOME/opt/bin"   # persistent: pinned binaries
# Postgres lives on the local disk. The synced home filesystem drops empty directories and resets
# permissions on restart, which breaks a Postgres data dir. The index tables are derived from
# Parquet, so db_init.sh rebuilds them if this directory is ever lost.
PGDATA=/var/lib/godseye/pg
mkdir -p "$DATA"/{redis,logs,secrets,media,index} "$OPT"
[[ -d /var/lib/godseye ]] || { sudo mkdir -p /var/lib/godseye && sudo chown "$USER:$USER" /var/lib/godseye; }

need_apt=()
dpkg -s ffmpeg >/dev/null 2>&1 || need_apt+=(ffmpeg)
dpkg -s redis-server >/dev/null 2>&1 || need_apt+=(redis-server)
dpkg -s "postgresql-$PG_MAJOR-pgvector" >/dev/null 2>&1 || need_apt+=("postgresql-$PG_MAJOR" "postgresql-$PG_MAJOR-pgvector")

if ((${#need_apt[@]})); then
  echo "[bootstrap] installing: ${need_apt[*]}"
  if [[ ! -f /etc/apt/sources.list.d/pgdg.list && ! -f /etc/apt/sources.list.d/pgdg.sources ]]; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q postgresql-common
    # We run our own cluster under $DATA/pg; don't let the package create a system one.
    sudo mkdir -p /etc/postgresql-common/createcluster.d
    echo "create_main_cluster = false" | sudo tee /etc/postgresql-common/createcluster.d/godseye.conf >/dev/null
    sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
  fi
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${need_apt[@]}"
fi

install_release() {  # name url checksum_url
  local name=$1 url=$2 sums_url=$3 tmp
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/pkg.tar.gz" "$url"
  curl -fsSL -o "$tmp/sums" "$sums_url"
  (cd "$tmp" && grep "$(basename "$url")" sums | awk '{print $1"  pkg.tar.gz"}' | sha256sum -c --quiet)
  tar -xzf "$tmp/pkg.tar.gz" -C "$tmp"
  install -m 0755 "$tmp/$name" "$OPT/$name"
  rm -rf "$tmp"
}

if ! "$OPT/mediamtx" --version 2>/dev/null | grep -q "$MEDIAMTX_VERSION"; then
  echo "[bootstrap] installing mediamtx $MEDIAMTX_VERSION"
  base="https://github.com/bluenviron/mediamtx/releases/download/$MEDIAMTX_VERSION"
  install_release mediamtx "$base/mediamtx_${MEDIAMTX_VERSION}_linux_amd64.tar.gz" "$base/checksums.sha256"
fi

if ! "$OPT/process-compose" version 2>/dev/null | grep -q "${PROCESS_COMPOSE_VERSION#v}"; then
  echo "[bootstrap] installing process-compose $PROCESS_COMPOSE_VERSION"
  base="https://github.com/F1bonacc1/process-compose/releases/download/$PROCESS_COMPOSE_VERSION"
  install_release process-compose "$base/process-compose_linux_amd64.tar.gz" "$base/process-compose_checksums.txt"
fi

if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
  echo "[bootstrap] initdb $PGDATA"
  "/usr/lib/postgresql/$PG_MAJOR/bin/initdb" -D "$PGDATA" -U postgres --auth=trust -E UTF8 >/dev/null
  cat >>"$PGDATA/postgresql.conf" <<EOF
listen_addresses = '127.0.0.1'
port = 5432
unix_socket_directories = '/tmp'
shared_buffers = 1GB
maintenance_work_mem = 1GB
EOF
fi

# MediaMTX publish password (phone -> Larix). Generated once, never committed.
PASS_FILE="$DATA/secrets/mediamtx_publish_pass"
[[ -s "$PASS_FILE" ]] || { openssl rand -hex 12 >"$PASS_FILE"; chmod 600 "$PASS_FILE"; }
sudo mkdir -p /var/lib/godseye/live && sudo chown "$USER" /var/lib/godseye/live   # recordings (local disk)
sed "s/__PUBLISH_PASS__/$(cat "$PASS_FILE")/" "$HOME/godsEye/infra/studio/mediamtx.yml" >"$DATA/mediamtx.yml"
chmod 600 "$DATA/mediamtx.yml"

echo "[bootstrap] ok"
