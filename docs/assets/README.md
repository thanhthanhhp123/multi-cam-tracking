# README assets

Media embedded in the top-level `README.md`. Committed here (small, part of the portfolio).

| File | What it shows |
|---|---|
| `cross_camera.gif` | Live dashboard during a replay: three people crossing `cam01 → cam02`, each keeping one Global ID; ground-plane map, camera footprints, trajectory trails, live "Đang theo dõi" panel |
| `dashboard.jpg` | Still frame of the same view |

## How it was captured

```bash
docker compose -f docker/compose.yml up -d redis
python -m tools.make_synthetic_fixture --out tests/fixtures/two_cam_walk.jsonl

python -m mct --source redis --db data/demo.db \
    --topology configs/cameras/topology.yaml \
    --homography-dir configs/demo/synthetic_homography --publish

MCT_DB_PATH=data/demo.db MCT_HOMOGRAPHY_DIR=configs/demo/synthetic_homography \
    uvicorn dashboard.app:app --port 8000

python -m tools.replay_metadata --fixture tests/fixtures/two_cam_walk.jsonl --speed 1.5
```

Browser window 1440×1024, page `http://localhost:8000`, recorded mid-replay.

## Notes / to improve

- Data is the **synthetic 2-camera fixture**, not the real DeepStream pipeline. It exercises the
  full path (tracklet builder → affinity → Hungarian → Global ID → SQLite → dashboard) with a
  controllable Re-ID model.
- `configs/demo/synthetic_homography/{cam01,cam02}.yaml` is a **stand-in calibration** for the
  synthetic scene so the ground-plane map renders — not a real camera homography.
- The dashboard UI strings are Vietnamese (per `CLAUDE.md` §8, docs are Vietnamese). An English
  UI pass is a possible follow-up if the portfolio audience needs it.
- Trajectory trails show a small zig-zag near the camera boundary — an artifact of the
  stand-in homography's foot-point mapping, not the engine.
- Blur any real faces before publishing if a shot ever comes from self-collected data
  (`CLAUDE.md` §11, privacy).
