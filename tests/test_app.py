import pytest

import main as app_module
from src.model import predict_model
from tests.conftest import SAMPLE_RAW_INPUT


@pytest.fixture
def client(monkeypatch, fitted_preprocessor, fitted_model):
    monkeypatch.setattr(predict_model, "_preprocessor", fitted_preprocessor)
    monkeypatch.setattr(predict_model, "_model", fitted_model)
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_endpoint_returns_a_price(client):
    response = client.post("/predict", json=SAMPLE_RAW_INPUT)

    assert response.status_code == 200
    body = response.get_json()
    assert isinstance(body["price"], (int, float))


def test_predict_endpoint_rejects_missing_fields(client):
    response = client.post("/predict", json={"model": "3 Series"})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_endpoint_rejects_non_json_body(client):
    response = client.post("/predict", data="not json", content_type="text/plain")

    assert response.status_code == 400
