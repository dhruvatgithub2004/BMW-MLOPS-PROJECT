---
title: BMW Price Predictor
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# BMW-MLOPS-PROJECT

End-to-end MLOps pipeline that predicts the resale price of a used BMW from its listing
attributes (model, year, mileage, fuel type, transmission, tax, mpg, engine size), served
behind a small Flask API.

- **Pipeline** (DVC): `data_ingestion → build_features → train` — see `dvc.yaml` and `PROJECT_PLAN.md`.
- **Experiment tracking**: MLflow, hosted on [DagsHub](https://dagshub.com/dhruvatgithub2004/BMW-MLOPS-PROJECT.mlflow).
- **Model**: RandomForestRegressor (R²≈0.94 on held-out test data), chosen over LinearRegression.
- **Serving**: Flask (`main.py`), containerized for deployment on Hugging Face Spaces (Docker SDK).

## API

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"model":"3 Series","year":2018,"transmission":"Automatic","mileage":15000,"fuelType":"Diesel","tax":145,"mpg":55.4,"engineSize":2.0}'
```

Returns `{"price": <predicted price>}`. `GET /health` returns `{"status": "ok"}`.

## Local development

```bash
pip install -r requirements.txt
python -m src.data.data_ingestion
python -m src.features.build_features
python -m src.model.train_model
python main.py            # dev server on :5000
# or
docker build -t bmw-price-api . && docker run -p 7860:7860 bmw-price-api
```

Run the test suite with `pytest`.
