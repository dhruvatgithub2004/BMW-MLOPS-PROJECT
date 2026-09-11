"""Feature engineering: fit the preprocessing pipeline on the training split and apply it
to both train and test splits, persisting the fitted preprocessor and the transformed arrays."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import logger
from src.utils.common import create_directories, read_yaml

CONFIG_PATH = Path("config.yaml")
TARGET_COLUMN = "price"


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that one-hot encodes categoricals and scales numerics."""
    categorical_cols = feature_df.select_dtypes(include="object").columns.tolist()
    numeric_cols = feature_df.select_dtypes(exclude="object").columns.tolist()

    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numeric columns: {numeric_cols}")

    return ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a dataframe into its feature columns and the target array."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].to_numpy()
    return X, y


def main():
    try:
        config = read_yaml(CONFIG_PATH)
        train_test_dir = Path(config.data_ingestion.train_test_dir)

        train_df = pd.read_csv(train_test_dir / "train.csv")
        test_df = pd.read_csv(train_test_dir / "test.csv")

        X_train, y_train = split_features_target(train_df)
        X_test, y_test = split_features_target(test_df)

        preprocessor = build_preprocessor(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        logger.info(f"Transformed train shape: {X_train_transformed.shape}, test shape: {X_test_transformed.shape}")

        processed_dir = Path(config.data_transformation.processed_data_dir)
        create_directories([processed_dir, Path(config.data_transformation.preprocessor_path).parent])

        np.save(processed_dir / "X_train.npy", X_train_transformed)
        np.save(processed_dir / "y_train.npy", y_train)
        np.save(processed_dir / "X_test.npy", X_test_transformed)
        np.save(processed_dir / "y_test.npy", y_test)
        logger.info(f"Processed train/test arrays saved to {processed_dir}")

        preprocessor_path = Path(config.data_transformation.preprocessor_path)
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    except Exception as e:
        logger.exception(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
