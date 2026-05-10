# Plant disease detector (CV pipeline)

EfficientNet-B3 classifier for 38 PlantVillage-style disease classes. This package lives beside the Plant Health Monitor web app; training artifacts default under `plant_disease_detector/`.

## Layout

| Path | Purpose |
|------|---------|
| `data/manifest.json` | Train/val/test sample index |
| `label_map.json` | Class id → name |
| `checkpoints/best.pt` | Training output |
| `results/metrics.json` | Evaluation metrics |
| `results/*.png` | Confusion matrix, per-class F1, Grad-CAM grid |
| `exports/` | ONNX + exported program (from `export.py`) |

## Environment variables

| Variable | Meaning |
|----------|---------|
| `PLANT_DISEASE_CHECKPOINT` | Path to `best.pt` |
| `PLANT_DISEASE_MANIFEST` | Path to `manifest.json` |
| `PLANT_DISEASE_LABEL_MAP` | Path to `label_map.json` |
| `SKIP_EVAL_GATE` / `CI` | If set, `evaluate.py` does not exit non-zero when top-1 &lt; 0.95 |

## Commands (run from repo root with `--project-root`, or `cd` this directory)

```bash
# Data: real Kaggle trees under data/raw/, or synthetic fallback for smoke/CI
python prepare_data.py --project-root /path/to/plant_project
python prepare_data.py --project-root /path/to/plant_project --synthetic-fallback

python scripts/sanity_check_model.py
python train.py                    # full schedule
python train.py --smoke          # tiny subset

SKIP_EVAL_GATE=1 python evaluate.py --checkpoint checkpoints/best.pt
python gradcam.py --image img1.jpg img2.jpg --output results/gradcam_grid.png --checkpoint checkpoints/best.pt
python predict.py --image sample.jpg
python export.py --checkpoint checkpoints/best.pt --out-dir exports
uvicorn api:app --host 0.0.0.0 --port 8010 --app-dir /path/to/plant_disease_detector
```

## Evaluation gate

The release gate is **top-1 accuracy ≥ 0.95** on the held-out test split. On random synthetic images this will fail (expected). With real PlantVillage data and sufficient training, re-run without `SKIP_EVAL_GATE` for a strict check.

## Tests

```bash
cd plant_disease_detector
pip install -r requirements.txt pytest pytest-cov httpx ruff
python -m pytest tests/ --cov=. --cov-fail-under=80
```

## Docker

The repo root `Dockerfile` installs these dependencies and sets `PYTHONPATH` to this folder so optional tooling can import the same modules. The default `docker-compose.yml` service builds from that Dockerfile.
