"""Data ingestion: load the raw BMW listings dataset and split it into train/test sets."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import logger
from src.utils.common import create_directories, read_yaml

CONFIG_PATH = Path("config.yaml")
PARAMS_PATH = Path("params.yaml")

EXPECTED_COLUMNS = [
    "model", "year", "price", "transmission",
    "mileage", "fuelType", "tax", "mpg", "engineSize",
]


def load_data(data_path: str) -> pd.DataFrame:
    """Load the raw BMW dataset from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded from {data_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {data_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file: {e}")
        raise


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure the dataframe has the columns the rest of the pipeline expects."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        logger.error(f"Missing expected columns: {missing}")
        raise KeyError(f"Missing expected columns: {missing}")


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray whitespace from string columns (the raw CSV has values like " 3 Series")
    so category strings are consistent between training data and future prediction requests."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def split_and_save(df: pd.DataFrame, test_size: float, random_state: int, output_dir: str) -> None:
    """Split the data into train/test sets and persist both to disk."""
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

    create_directories([output_dir])
    train_path = Path(output_dir) / "train.csv"
    test_path = Path(output_dir) / "test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    logger.info(f"Train data {train_data.shape} saved to {train_path}")
    logger.info(f"Test data {test_data.shape} saved to {test_path}")


def main():
    try:
        config = read_yaml(CONFIG_PATH)
        params = read_yaml(PARAMS_PATH)

        df = load_data(config.data_source.raw_data_path)
        validate_columns(df)
        df = clean_categoricals(df)

        split_and_save(
            df,
            test_size=params.data_ingestion.test_size,
            random_state=params.data_ingestion.random_state,
            output_dir=config.data_ingestion.train_test_dir,
        )
    except Exception as e:
        logger.exception(f"Data ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
