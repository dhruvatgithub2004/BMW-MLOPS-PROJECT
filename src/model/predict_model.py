"""Inference: apply the fitted preprocessor + model to raw feature input to predict a price."""
from pathlib import Path

import joblib
import pandas as pd

from src import logger
from src.utils.common import read_yaml

CONFIG_PATH = Path("config.yaml")

# Order doesn't matter (the preprocessor selects columns by name), but every field is required.
REQUIRED_FIELDS = ["model", "year", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize"]

_preprocessor = None
_model = None


def _load_artifacts():
    """Lazily load and cache the fitted preprocessor and model."""
    global _preprocessor, _model
    if _preprocessor is None or _model is None:
        config = read_yaml(CONFIG_PATH)
        _preprocessor = joblib.load(config.data_transformation.preprocessor_path)
        _model = joblib.load(config.model_trainer.model_path)
        logger.info("Preprocessor and model loaded for inference")
    return _preprocessor, _model


def predict(raw_input: dict) -> float:
    """Predict the price of a BMW listing from its raw feature values.

    `raw_input` must contain every field in REQUIRED_FIELDS, e.g.:
        {"model": "3 Series", "year": 2018, "transmission": "Automatic",
         "mileage": 15000, "fuelType": "Diesel", "tax": 145, "mpg": 55.4, "engineSize": 2.0}
    """
    missing = set(REQUIRED_FIELDS) - set(raw_input)
    if missing:
        raise KeyError(f"Missing required fields: {sorted(missing)}")

    preprocessor, model = _load_artifacts()
    df = pd.DataFrame([{field: raw_input[field] for field in REQUIRED_FIELDS}])
    X = preprocessor.transform(df)
    prediction = float(model.predict(X)[0])
    logger.info(f"Predicted price {prediction:.2f} for input {raw_input}")
    return prediction


def main():
    sample = {
        "model": "3 Series",
        "year": 2018,
        "transmission": "Automatic",
        "mileage": 15000,
        "fuelType": "Diesel",
        "tax": 145,
        "mpg": 55.4,
        "engineSize": 2.0,
    }
    print(f"Predicted price: {predict(sample):.2f}")


if __name__ == "__main__":
    main()
