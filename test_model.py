import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "project_config.json") as f:
    config = json.load(f)

model = joblib.load(MODEL_DIR / "final_kmeans_model.pkl")
scaler = joblib.load(MODEL_DIR / "final_scaler.pkl")

FEATURES = config["features"]
SEGMENTS = config["behavior_segments"]


def test_model_artifacts_load():
    assert len(FEATURES) == 20
    assert len(SEGMENTS) == 4
    assert hasattr(model, "predict")
    assert hasattr(scaler, "transform")


def test_prediction_output():
    sample = pd.DataFrame(
        [[0.0] * len(FEATURES)],
        columns=FEATURES
    )
    scaled = scaler.transform(sample)
    predictions = model.predict(scaled)

    assert len(predictions) == 1
    assert int(predictions[0]) in range(4)
    assert str(int(predictions[0])) in SEGMENTS


def test_multiple_students():
    sample = pd.DataFrame(
        [[0.0] * len(FEATURES), [1.0] * len(FEATURES)],
        columns=FEATURES
    )
    predictions = model.predict(scaler.transform(sample))

    assert len(predictions) == 2
    assert all(int(p) in range(4) for p in predictions)
