"""Gradio demo for the BMW price-prediction model, deployed as a Hugging Face Space.

Self-contained on purpose: loads models/*.pkl relative to this file rather than importing
the `src` package, so this directory can be copied standalone into a fresh Space repo.
"""
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"
preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
model = joblib.load(MODEL_DIR / "model.pkl")

# Categories and ranges pulled from the actual training data (data/raw/bmw.csv).
MODEL_CHOICES = [
    "1 Series", "2 Series", "3 Series", "4 Series", "5 Series", "6 Series",
    "7 Series", "8 Series", "M2", "M3", "M4", "M5", "M6", "X1", "X2", "X3",
    "X4", "X5", "X6", "X7", "Z3", "Z4", "i3", "i8",
]
TRANSMISSION_CHOICES = ["Automatic", "Manual", "Semi-Auto"]
FUEL_CHOICES = ["Diesel", "Electric", "Hybrid", "Other", "Petrol"]


def predict_price(model_name, year, transmission, mileage, fuel_type, tax, mpg, engine_size):
    df = pd.DataFrame([{
        "model": model_name,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size,
    }])
    X = preprocessor.transform(df)
    price = float(model.predict(X)[0])
    return f"£{price:,.2f}"


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(MODEL_CHOICES, value="3 Series", label="Model"),
        gr.Slider(1996, 2020, value=2018, step=1, label="Year"),
        gr.Dropdown(TRANSMISSION_CHOICES, value="Automatic", label="Transmission"),
        gr.Number(value=15000, label="Mileage"),
        gr.Dropdown(FUEL_CHOICES, value="Diesel", label="Fuel type"),
        gr.Number(value=145, label="Road tax"),
        gr.Number(value=55.4, label="MPG"),
        gr.Slider(0.0, 6.6, value=2.0, step=0.1, label="Engine size (litres)"),
    ],
    outputs=gr.Textbox(label="Predicted price"),
    title="BMW Price Predictor",
    description=(
        "Predicts the resale price of a used BMW from its listing attributes. "
        "RandomForestRegressor trained on UK used-car listings (R²≈0.94). "
        "Part of an end-to-end MLOps pipeline (DVC + MLflow/DagsHub) — "
        "see https://github.com/dhruvatgithub2004/BMW-MLOPS-PROJECT"
    ),
)

if __name__ == "__main__":
    demo.launch()
