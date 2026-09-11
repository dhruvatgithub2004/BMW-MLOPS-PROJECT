import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Small synthetic dataset shaped like data/raw/bmw.csv, used so tests don't depend on
# the real (DVC-tracked, possibly-not-pulled) preprocessor/model artifacts.
SAMPLE_ROWS = pd.DataFrame({
    "model": ["3 Series", "5 Series", "3 Series", "1 Series"],
    "year": [2018, 2016, 2019, 2015],
    "transmission": ["Automatic", "Manual", "Automatic", "Manual"],
    "mileage": [15000, 40000, 8000, 60000],
    "fuelType": ["Diesel", "Petrol", "Diesel", "Petrol"],
    "tax": [145, 125, 145, 30],
    "mpg": [55.4, 45.6, 55.4, 60.1],
    "engineSize": [2.0, 2.0, 2.0, 1.5],
})
SAMPLE_PRICES = np.array([21000.0, 15000.0, 23000.0, 8000.0])

SAMPLE_RAW_INPUT = {
    "model": "3 Series",
    "year": 2018,
    "transmission": "Automatic",
    "mileage": 15000,
    "fuelType": "Diesel",
    "tax": 145,
    "mpg": 55.4,
    "engineSize": 2.0,
}


@pytest.fixture
def fitted_preprocessor():
    categorical_cols = ["model", "transmission", "fuelType"]
    numeric_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
    preprocessor = ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ])
    preprocessor.fit(SAMPLE_ROWS)
    return preprocessor


@pytest.fixture
def fitted_model(fitted_preprocessor):
    X = fitted_preprocessor.transform(SAMPLE_ROWS)
    return LinearRegression().fit(X, SAMPLE_PRICES)
