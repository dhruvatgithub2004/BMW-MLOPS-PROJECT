import numpy as np
import pandas as pd

from src.features.build_features import build_preprocessor


def make_feature_df() -> pd.DataFrame:
    return pd.DataFrame({
        "model": ["3 Series", "5 Series", "3 Series"],
        "year": [2018, 2016, 2019],
        "transmission": ["Automatic", "Manual", "Automatic"],
        "mileage": [15000, 40000, 8000],
        "fuelType": ["Diesel", "Petrol", "Diesel"],
        "tax": [145, 125, 145],
        "mpg": [55.4, 45.6, 55.4],
        "engineSize": [2.0, 2.0, 2.0],
    })


def test_build_preprocessor_routes_columns_by_dtype():
    preprocessor = build_preprocessor(make_feature_df())

    ohe_cols = next(cols for name, _, cols in preprocessor.transformers if name == "ohe")
    num_cols = next(cols for name, _, cols in preprocessor.transformers if name == "num")

    assert set(ohe_cols) == {"model", "transmission", "fuelType"}
    assert set(num_cols) == {"year", "mileage", "tax", "mpg", "engineSize"}


def test_transform_shape_matches_row_count():
    df = make_feature_df()
    preprocessor = build_preprocessor(df)

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == len(df)
    assert np.all(np.isfinite(transformed))


def test_transform_ignores_unseen_categories_without_raising():
    df = make_feature_df()
    preprocessor = build_preprocessor(df)
    preprocessor.fit(df)

    unseen = df.copy()
    unseen.loc[0, "model"] = "8 Series"  # not present in the fit data

    transformed = preprocessor.transform(unseen)

    assert transformed.shape == (len(df), preprocessor.transform(df).shape[1])
    assert np.all(np.isfinite(transformed))
