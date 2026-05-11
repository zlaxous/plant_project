# Plant Disease Detection Project

This repository contains a complete plant disease detection project with:

- a production-focused transfer learning pipeline (`plant_disease_detector/`) using EfficientNet-B3,
- a custom CNN-from-scratch implementation (`disscution_project/CNN_from_scratch.py`),
- a Streamlit dashboard to compare both models side by side (`disscution_project/streamlit_app.py`),
- helper tooling to build dataset manifests and label maps (`build_manifest.py`).

The dataset setup targets PlantVillage-style classes (38 classes).

## Project Structure

```text
plant_project/
|- README.md
|- build_manifest.py
|- plant_disease_detector/
|  |- train.py
|  |- evaluate.py
|  |- predict.py
|  |- prepare_data.py
|  |- export.py
|  |- gradcam.py
|  |- model.py
|  |- dataset.py
|  |- paths.py
|  |- pyproject.toml
|  |- requirements.txt
|  |- data/
|  |- checkpoints/
|  |- results/
|  |- tests/
|- disscution_project/
   |- CNN_from_scratch.py
   |- streamlit_app.py
   |- checkpoints/
```

## Requirements

- Python 3.11+
- pip
- (Optional but recommended) NVIDIA GPU with CUDA for faster training

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r .\plant_disease_detector\requirements.txt
pip install pytest pytest-cov streamlit opencv-python
```

If you prefer package metadata:

```powershell
pip install -e .\plant_disease_detector
```

## Data Preparation

Prepare the dataset and metadata:

```powershell
python .\plant_disease_detector\prepare_data.py --project-root .
```

For quick smoke/testing runs without full real data:

```powershell
python .\plant_disease_detector\prepare_data.py --project-root . --synthetic-fallback
```

You can also generate manifest/label files from `_all_images.txt` with:

```powershell
python .\build_manifest.py
```

## Training

### Transfer Learning (EfficientNet-B3)

```powershell
python .\plant_disease_detector\train.py
```

Quick smoke run:

```powershell
python .\plant_disease_detector\train.py --smoke
```

Resume interrupted training:

```powershell
python .\plant_disease_detector\train.py --resume
```

### CNN From Scratch

```powershell
python .\disscution_project\CNN_from_scratch.py
```

Quick smoke run:

```powershell
python .\disscution_project\CNN_from_scratch.py --smoke
```

Resume interrupted training:

```powershell
python .\disscution_project\CNN_from_scratch.py --resume
```

## Evaluation and Inference

Evaluate a trained checkpoint:

```powershell
python .\plant_disease_detector\evaluate.py --checkpoint .\plant_disease_detector\checkpoints\best.pt
```

Predict on a single image:

```powershell
python .\plant_disease_detector\predict.py --image .\path\to\leaf.jpg --topk 5
```

Generate Grad-CAM visualizations:

```powershell
python .\plant_disease_detector\gradcam.py --image .\path\to\leaf.jpg --checkpoint .\plant_disease_detector\checkpoints\best.pt
```

Export models for deployment (TorchScript/ONNX):

```powershell
python .\plant_disease_detector\export.py --checkpoint .\plant_disease_detector\checkpoints\best.pt --out-dir .\plant_disease_detector\exports
```

## Streamlit Comparison Dashboard

Run the interactive UI that compares:

- Scratch CNN predictions
- EfficientNet-B3 predictions
- Grad-CAM maps
- confidence, severity, and inference-time metrics

```powershell
streamlit run .\disscution_project\streamlit_app.py
```

Then open the local URL shown by Streamlit in your browser.

## Tests

Run the detector test suite:

```powershell
cd .\plant_disease_detector
python -m pytest .\tests\ --cov=. --cov-fail-under=80
```

## Notes

- Main pipeline docs are in `plant_disease_detector/README.md`.
- Default checkpoints are expected under `plant_disease_detector/checkpoints/` and `disscution_project/checkpoints/`.
- If class names or split metadata look wrong, regenerate manifest and label map before retraining.
