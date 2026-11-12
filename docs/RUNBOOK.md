# Running godsEye

How to run the MVP day to day: where each piece runs, what it costs, and the commands. For what godsEye is,
see the [README](../README.md); for how it works, [ARCHITECTURE.md](../ARCHITECTURE.md).

## Where everything runs

| Piece | Runs on | Costs | Lives in |
|---|---|---|---|
| Code editing, the web console (`apps/web`, Next.js), your browser | **Your Mac** | free | this repo |
| Postgres + pgvector, Redis, MediaMTX (video server), search API (FastAPI) | **Lightning AI Studio** `godsEye` (free CPU machine; switch to an **L4 GPU** for live sessions) | CPU free; L4 $0.79/h from Lightning credits; storage over 10 GB at $0.10/GB-month | `~/godsEye` (code, synced from the Mac), `~/data` (footage, index, logs, secrets) |
| Batch indexing (YOLO26 + tracking + SigLIP 2 embeddings, then the body-part pass with YOLO26-pose) | **Modal** (serverless L4 GPUs, one per camera) | ~$1.60 + ~$0.60 (parts) for the 6.5-hour spike slice; $30/month free credit, spend limit $0 | Modal Volume `godseye-index` |
| Query planner and result verifier | **Anthropic API** (Haiku 5.5; Sonnet 5.5 checks body-part details like tattoos or glasses) | ~1-4 cents per search; ~10 cents for a detail search | your API key |
| Footage | **MEVA public S3 bucket** (us-east-1) | free | `s3://mevadata-public-01` |
| Source code | **GitHub** `newacronym/godsEye` | free | |

Nothing on the Studio is reachable from the internet. Your Mac reaches it through one SSH tunnel.

## One-time setup on the Mac

```bash
cp infra/ssh_config.example infra/ssh_config   # then fill in your Studio's SSH user
lightning login             # Lightning CLI (already installed), lets you start/stop the Studio from here
modal token new             # only if `modal profile current` doesn't show your account
```

## Everyday use

All `make` commands run from the repository root on the Mac.

**Start the Studio** (the free CPU Studio stops on its own after ~4 h or when idle):
```bash
lightning studio start --teamspace general --name godseye
```
or press **Start** on lightning.ai. On start, the Studio launches Postgres, Redis and MediaMTX by
itself and rebuilds the database from the index files (1-3 minutes; the first minutes after a start
are slow because Lightning loads files from its storage on first access).

**Check it:**
```bash
make status                 # services and their state
make logs                   # last 50 lines of the service log
```

**Open the console** (needs `make tunnel` running; keep this in its own terminal tab):
```bash
cd apps/web && npm install && npm run dev     # first time: npm install; afterwards just npm run dev
```
Then open **http://localhost:3000**:
- **Ask**: describe what you're looking for in plain words ("a man in a red Nike t-shirt", "someone
  with a tattoo on his right hand", "a black SUV"), or attach a **photo of a person** (button in the
  question bar, or paste/drop one) to find where that person appears. Results stream in: first results,
  the interpretation (each condition and where it is checked), then verified and possible matches with
  a per-condition check. Pick *Search in* to limit it to one uploaded video. Searches are shareable
  links (`/?q=...`).
- **Videos**: drop a video file to upload it. It is prepared on the Studio, indexed on a Modal GPU and
  searchable a few minutes later (a 90-second clip took ~5 minutes end to end); the page shows each
  step. Remove an upload with its trash button.
- **Cameras**: live streams.

Stop with Ctrl-C.

The search API itself runs on the Studio as the `api` service (started automatically with the
others; `make status` lists it) and is reachable from the Mac at http://localhost:8000.

**Watch video** (keep this running in its own terminal tab):
```bash
make tunnel
```
It reconnects by itself when Lightning's SSH gateway drops the connection ("tunnel closed;
reconnecting"); Ctrl-C stops it. If it keeps printing "Operation timed out", the gateway or the
Studio is down: check the Studio on lightning.ai.
Then, in Chrome on the Mac:
- `http://localhost:8889/<stream>/`: live view (WebRTC, under 1 s delay)
- `http://localhost:8888/<stream>/`: fallback view (HLS, 2-4 s delay)
- `open -a "Google Chrome" scripts/wall.html`: 2×2 wall of the MEVA test cameras

**Replay MEVA as live cameras** (for testing alerts without a real camera):
```bash
make fake-live              # G330 G339 G421 G639 as meva-<cam> (720p) for the browser
make fake-live-status
make fake-live-stop
```

**Phone as a camera** (Android/iOS, Larix Broadcaster, same Wi-Fi as the Mac, `make tunnel` running):
`make publish-pass` copies the stream password to the Mac's clipboard (`pbpaste` shows it), then in
Larix set the URL to `rtmp://<Mac's Wi-Fi IP>:1935/live/phone1?user=publisher&pass=<password>` (H.264,
720p, ~2.5 Mbps, keyframe every 1 s). The phone shows up on the console's Cameras page.

**Search a live stream**: on the Cameras page press **Index live** on the phone (or a replayed MEVA
camera). A Modal L4 starts in ~35 s, then indexes 5 frames a second about a second behind live; press
**Search** (or pick *Live: phone1* in Ask) to search it while it runs. It costs ~$0.80/hour of Modal
credit, stops a minute after you press Stop or the stream ends, and stops by itself after 2 hours.
Every stream is recorded on the Studio for 6 hours, so live results replay like any video.

**Anthropic key** (stored on the Studio, overrides Lightning's secret, no restart needed):
```bash
make set-anthropic-key
```

## Stopping things (and spending nothing)

```bash
make fake-live-stop                                        # replayed cameras
make down                                                  # services on the Studio
lightning studio stop --teamspace general --name godseye  # the Studio itself
```
- The CPU Studio is free, but an **L4 Studio bills by the second**. Switch back to CPU when done:
  `lightning studio switch --teamspace general --name godseye --machine CPU`
- Stopping the Studio loses nothing: footage, index files, code and secrets are kept, and the
  database is rebuilt from the index files on the next start.
- Modal only bills while a job runs. To see or stop running jobs: `modal app list`, `modal app stop <app-id>`.

## Batch indexing (Modal)

```bash
modal run services/indexer/index_app.py --cameras G330   # one camera
modal run services/indexer/index_app.py                  # whole spike slice (13 cameras in parallel)
```
Then on the Studio (`make ssh`): `cd ~/data/index && ~/.local/bin/modal volume get --force godseye-index spike .`
and `cd ~/godsEye && uv run services/indexer/load.py --run spike`.

**Uploads from the console** call the deployed indexer, so deploy it once (and again after changing
`services/indexer/`; a deployed app costs nothing while idle):
```bash
modal deploy services/indexer/index_app.py
```

**Body-part pass** (hands, heads, torsos, legs, feet of people close enough to show detail; needs the first pass):
```bash
modal run services/indexer/index_app.py --parts          # ~10 min, writes parts.parquet next to tracks.parquet
```
Fetch and load the same way (the loader picks up `parts.parquet` when present). Questions about small details
("a tattoo on his right hand", "someone wearing glasses", "a logo on their shirt") then search these close-ups.

## Evaluation

On the Studio (`make ssh`, then `cd ~/godsEye`):
```bash
PYTHONPATH=services uv run scripts/spike/eval_retrieval.py                     # retrieval only
PYTHONPATH=services uv run scripts/spike/eval_retrieval.py --verify --tag v3   # + Haiku planner/verifier
```
Per-candidate verdicts go to `~/data/logs/eval_candidates_<tag>.jsonl`.

Appearance questions MEVA doesn't label (clothing, accessories, vehicles, combinations) are judged by eye:
```bash
PYTHONPATH=services uv run scripts/spike/eval_general.py      # 28 questions; contact sheets in ~/data/logs/general_eval
```
Judgments live in `~/data/logs/general_eval/judgments.json`; results nobody judged yet are listed.

Details MEVA doesn't label (tattoos, glasses) are measured against mined ground truth:
```bash
PYTHONPATH=services uv run scripts/spike/mine_details.py --part hand --detail "a tattoo" --tag tattoo      # screen every view
PYTHONPATH=services uv run scripts/spike/eval_details.py --tag tattoo_right                                # after confirming by eye
PYTHONPATH=services uv run scripts/spike/show_parts.py hand "a hand with a tattoo"                          # what retrieval sees
PYTHONPATH=services uv run scripts/spike/show_detail_evidence.py hand "a hand with a tattoo" <track id>     # what the verifier sees
```

## What's where on the Studio

| Path | What |
|---|---|
| `~/godsEye` | code (synced from the Mac by `make sync`; don't edit here) |
| `~/data/meva/raw`, `~/data/meva/proxy` | MEVA 1080p originals, 720p viewing copies |
| `~/data/meva/annotations` | MEVA labels (`events.csv`, `boxes.csv.gz`) |
| `~/data/index/<run>/<camera>/` | index Parquet from Modal (the source of truth for the database); uploads are run `uploads` |
| `~/data/uploads/<camera>/` | an uploaded video: `source.json` (its record), `original.mp4` (1080p, indexed), `proxy.mp4` (720p, for the browser), `thumb.jpg` |
| `~/data/queries/` | photos people searched by |
| `/var/lib/godseye/pg` | Postgres data (local disk; rebuilt from Parquet on each start) |
| `~/data/secrets` | MediaMTX publish password, Anthropic key |
| `~/data/logs` | service, startup, evaluation logs |
