# README assets

Screenshots / GIFs embedded in the top-level `README.md`. Commit the image files here
(they are small and part of the portfolio); do not commit raw video.

## Shots to capture

Run the demo data path first:

```bash
make up
make engine-fixture FIXTURE=tests/fixtures/ds_4cam_reid_realtime.jsonl DB=data/demo.db
MCT_DB_PATH=data/demo.db make dashboard          # http://localhost:8000
make replay FIXTURE=tests/fixtures/ds_4cam_reid_realtime.jsonl   # separate shell
```

| File | What it shows | How |
|---|---|---|
| `dashboard.png` | Ground-plane map with camera footprints + current positions, one Global ID selected so its trajectory is highlighted | Full-window PNG of `http://localhost:8000` while `make replay` is running |
| `cross_camera.gif` | The same person keeping one Global ID across 3+ cameras | Screen-record the map during a replay, trim to ~6–10 s, export GIF (e.g. with ScreenToGif on Windows), keep width ≤ 1200 px |

Optional extras: `trajectory.png` (the per-Global-ID trajectory table / lookup panel),
`architecture.png` (rendered from the mermaid diagram in the README).

Blur any real faces before publishing if a shot ever comes from self-collected data
(CLAUDE.md §11, privacy).
