"""Flask API serving BMW price predictions from the trained model."""
from flask import Flask, jsonify, request
from flask_cors import CORS

from src import logger
from src.model.predict_model import REQUIRED_FIELDS, predict

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict_price():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Request body must be JSON"), 400

    missing = set(REQUIRED_FIELDS) - set(payload)
    if missing:
        return jsonify(error=f"Missing required fields: {sorted(missing)}"), 400

    try:
        price = predict(payload)
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        return jsonify(error="Prediction failed"), 500

    return jsonify(price=price)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
