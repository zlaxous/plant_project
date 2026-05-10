# Plant project

Monorepo for **Plant Health Monitor** (FastAPI + React) and the **plant disease detector** CV pipeline under `plant_disease_detector/`.

## Quick start (local)

- **Web app**: see `backend/` and `frontend/`; `docker compose up --build` uses the root `Dockerfile` (includes detector dependencies and mounts `plant_disease_detector/data` + `checkpoints`).
- **CV pipeline**: see [`plant_disease_detector/README.md`](plant_disease_detector/README.md). Optional: `./setup.sh` creates a venv and installs Python deps.

## Docker

- **Default**: `docker compose` → root `Dockerfile` (single `app` service on port 8000).
- **Legacy slim image**: `docker build -f backend/Dockerfile -t plant-backend .`
