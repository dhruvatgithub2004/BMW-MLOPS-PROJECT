---
title: BMW Price Predictor
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# BMW Price Predictor

Predicts the resale price of a used BMW from its listing attributes (model, year, mileage,
transmission, fuel type, tax, mpg, engine size). RandomForestRegressor, R²≈0.94 on held-out
test data.

Part of an end-to-end MLOps pipeline (DVC pipeline, MLflow experiment tracking on DagsHub,
pytest suite) — source at https://github.com/dhruvatgithub2004/BMW-MLOPS-PROJECT. This Space
is just the serving layer; training/tracking/versioning happen upstream in that repo.
