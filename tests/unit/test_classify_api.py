"""Tests for POST /v1/classify and the /demo page.

A fake model stands in for the real one, so these tests run instantly and don't need a
trained model file. FastAPI's dependency overrides swap it in for the duration of a test.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.routes_triage import classifier_dependency
from app.main import app
from app.services import classify as classify_module
from app.services.classify import ClassifierService, ModelNotAvailableError, latest_model_path


class FakeModel:
    """Returns a fixed answer shaped like CalibratedClassifier.predict()."""

    def __init__(self, options):
        self.options = options

    def predict(self, texts):
        top = self.options[0]
        return [{
            "top": top["chapter"], "top_name": top["name"], "probability": top["probability"],
            "prediction_set": self.options, "coverage_target": 0.95,
        } for _ in texts]


SURE = [{"chapter": "32", "name": "Landing Gear", "probability": 0.93}]
UNSURE = [
    {"chapter": "32", "name": "Landing Gear", "probability": 0.55},
    {"chapter": "29", "name": "Hydraulic Power", "probability": 0.35},
]


@pytest.fixture
def client_with(request):
    def make(options):
        service = ClassifierService(FakeModel(options), "test-v1", confidence_threshold=0.85)
        app.dependency_overrides[classifier_dependency] = lambda: service
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_confident_prediction(client_with):
    r = client_with(SURE).post("/v1/classify", json={"text": "MLG ACTUATOR LEAK"})
    assert r.status_code == 200
    body = r.json()
    assert body["top"] == {"chapter": "32", "name": "Landing Gear", "probability": 0.93}
    assert body["confident"] is True
    assert body["model_version"] == "test-v1"
    assert "Advisory only" in body["advisory"]


def test_unsure_prediction_lists_alternatives_and_is_not_confident(client_with):
    body = client_with(UNSURE).post("/v1/classify", json={"text": "FLUID LEAK NEAR WHEEL WELL"}).json()
    assert [o["chapter"] for o in body["prediction_set"]] == ["32", "29"]
    assert body["confident"] is False  # two options -> not safe for the fast path


def test_high_probability_but_below_threshold_is_not_confident(client_with):
    almost = [{"chapter": "33", "name": "Lights", "probability": 0.80}]
    body = client_with(almost).post("/v1/classify", json={"text": "LIGHT INOP"}).json()
    assert body["confident"] is False  # 0.80 < 0.85 threshold


@pytest.mark.parametrize("text", ["", "  ", "ab", " \n a \n "])
def test_too_short_or_blank_text_is_rejected(client_with, text):
    r = client_with(SURE).post("/v1/classify", json={"text": text})
    assert r.status_code == 422


def test_too_long_text_is_rejected(client_with):
    r = client_with(SURE).post("/v1/classify", json={"text": "A" * 5001})
    assert r.status_code == 422


def test_missing_text_field_is_rejected(client_with):
    assert client_with(SURE).post("/v1/classify", json={}).status_code == 422


def test_no_trained_model_gives_clear_503(monkeypatch, tmp_path):
    monkeypatch.setattr(classify_module.settings, "models_dir", str(tmp_path))
    classify_module.get_classifier.cache_clear()
    try:
        r = TestClient(app).post("/v1/classify", json={"text": "MLG ACTUATOR LEAK"})
        assert r.status_code == 503
        assert "train_calibrated" in r.json()["detail"]
    finally:
        classify_module.get_classifier.cache_clear()


def test_latest_model_path_picks_newest(tmp_path):
    for name in ["ata_calibrated_20260101-000000.joblib", "ata_calibrated_20261114-021636.joblib"]:
        (tmp_path / name).touch()
    assert latest_model_path(tmp_path).name == "ata_calibrated_20261114-021636.joblib"
    with pytest.raises(ModelNotAvailableError):
        latest_model_path(tmp_path / "empty")


def test_demo_page_is_served():
    r = TestClient(app).get("/demo")
    assert r.status_code == 200
    assert "SkyTriage" in r.text and "/v1/classify" in r.text


def test_classify_endpoint_is_documented():
    assert "/v1/classify" in TestClient(app).get("/openapi.json").json()["paths"]


def test_unreadable_model_file_gives_clear_503_not_a_crash(monkeypatch, tmp_path):
    (tmp_path / "ata_calibrated_20260101-000000.joblib").write_text("not a real model")
    monkeypatch.setattr(classify_module.settings, "models_dir", str(tmp_path))
    classify_module.get_classifier.cache_clear()
    try:
        r = TestClient(app).post("/v1/classify", json={"text": "MLG ACTUATOR LEAK"})
        assert r.status_code == 503
        assert "Could not load" in r.json()["detail"]
    finally:
        classify_module.get_classifier.cache_clear()