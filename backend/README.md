---
title: AI Skin Disease Prediction
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI-Assisted Skin Disease Prediction — Backend API

EfficientNetV2-S + MSAM (Multi-Scale Attention Module) skin disease
classifier, served as a JSON API. Paired with a separate static frontend
deployed on Vercel.

## Endpoints

- `GET /` — health check
- `POST /predict` — multipart form upload, field name `image` (PNG/JPG).
  Returns JSON:
  ```json
  {
    "top_results": [{"label": "...", "confidence": 87.32}, ...],
    "top_prediction": {"label": "...", "confidence": 87.32},
    "clinical_info": {"description": "...", "symptoms": "...", "causes": "...", "treatment": "...", "advice": "..."},
    "disclaimer": "...",
    "gradcam_before": "data:image/png;base64,...",
    "gradcam_after": "data:image/png;base64,..."
  }
  ```

## Before pushing to this Space

Copy your trained model to `model/best_balanced.keras` and track it with
Git LFS (see the deployment README in the project root for full steps).

Research/educational use only — not a diagnostic medical device.
