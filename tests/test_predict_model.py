import pytest

from src.model import predict_model
from tests.conftest import SAMPLE_RAW_INPUT


@pytest.fixture(autouse=True)
def patch_artifacts(monkeypatch, fitted_preprocessor, fitted_model):
    """Point predict_model at fixture artifacts instead of the real (DVC-tracked) ones."""
    monkeypatch.setattr(predict_model, "_preprocessor", fitted_preprocessor)
    monkeypatch.setattr(predict_model, "_model", fitted_model)


def test_predict_is_deterministic_for_the_same_input():
    first = predict_model.predict(SAMPLE_RAW_INPUT)
    second = predict_model.predict(SAMPLE_RAW_INPUT)

    assert first == second
    assert isinstance(first, float)


def test_predict_raises_on_missing_fields():
    incomplete_input = {"model": "3 Series", "year": 2018}

    with pytest.raises(KeyError):
        predict_model.predict(incomplete_input)
