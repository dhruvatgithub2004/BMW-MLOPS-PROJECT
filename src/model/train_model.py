"""Model training: fit candidate regressors on the engineered features, track each run
with MLflow (backed by DagsHub), and persist the best-performing model."""
from pathlib import Path

import dagshub
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import logger
from src.utils.common import create_directories, read_yaml, save_json

CONFIG_PATH = Path("config.yaml")
PARAMS_PATH = Path("params.yaml")


def load_processed_data(processed_dir: str):
    """Load the train/test feature arrays produced by build_features.py."""
    processed_dir = Path(processed_dir)
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_pred) -> dict:
    """Compute the regression metrics used to compare candidate models."""
    return {
        "r2_score": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_and_log(name, model, X_train, y_train, X_test, y_test, params: dict) -> dict:
    """Fit `model`, log its params/metrics/artifact to MLflow, and return its test metrics."""
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        logger.info(f"{name} -> {metrics}")
    return metrics


def setup_mlflow(config) -> None:
    """Point MLflow at the DagsHub-hosted tracking server for this repo."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    dagshub.init(
        repo_owner=config.mlflow.dagshub_repo_owner,
        repo_name=config.mlflow.dagshub_repo_name,
        mlflow=True,
    )
    mlflow.set_experiment(config.mlflow.experiment_name)


def main():
    try:
        config = read_yaml(CONFIG_PATH)
        params = read_yaml(PARAMS_PATH)

        setup_mlflow(config)

        X_train, y_train, X_test, y_test = load_processed_data(
            config.data_transformation.processed_data_dir
        )

        rf_params = dict(params.model_trainer.random_forest)
        candidates = {
            "Linear Regression": (LinearRegression(), {}),
            "Random Forest Regressor": (RandomForestRegressor(**rf_params), rf_params),
        }

        results = {
            name: {
                "model": model,
                "metrics": train_and_log(name, model, X_train, y_train, X_test, y_test, log_params),
            }
            for name, (model, log_params) in candidates.items()
        }

        best_name = max(results, key=lambda n: results[n]["metrics"]["r2_score"])
        best_model = results[best_name]["model"]
        logger.info(f"Best model: {best_name} ({results[best_name]['metrics']})")

        create_directories([config.model_trainer.model_dir])
        model_path = Path(config.model_trainer.model_path)
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")

        metrics_path = Path(config.reports.metrics_path)
        create_directories([metrics_path.parent])
        save_json(metrics_path, {
            "best_model": best_name,
            "best_model_metrics": results[best_name]["metrics"],
            "all_models": {name: r["metrics"] for name, r in results.items()},
        })
    except Exception as e:
        logger.exception(f"Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()
