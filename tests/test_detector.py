"""Unit tests for detector model integration and output formatting."""

from __future__ import annotations

import numpy as np
import pytest

import detector
from config import MODEL_PATH


class FakeBox:
    """Minimal YOLO box-like object for test doubles."""

    def __init__(self, class_id: int, confidence: float, bbox: list[float]) -> None:
        self.cls = np.array([class_id], dtype=float)
        self.conf = np.array([confidence], dtype=float)
        self.xyxy = np.array([bbox], dtype=float)


class FakeResult:
    """Minimal YOLO result-like object for test doubles."""

    def __init__(self, boxes: list[FakeBox], names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class FakeYOLO:
    """Test double for Ultralytics YOLO model."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def to(self, device: str) -> "FakeYOLO":
        del device
        return self

    def predict(self, source: str | None, conf: float, iou: float, verbose: bool) -> list[FakeResult]:
        if source is None:
            raise ValueError("Invalid input source")

        if source == "empty.jpg":
            return []

        del conf, iou, verbose
        return [
            FakeResult(
                boxes=[
                    FakeBox(0, 0.91, [10.0, 20.0, 60.0, 150.0]),
                    FakeBox(1, 0.83, [100.0, 18.0, 145.0, 146.0]),
                ],
                names={0: "bottle", 1: "cup"},
            )
        ]


def test_model_loads_with_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detector should initialize with the configured default model path."""
    monkeypatch.setattr(detector, "YOLO", FakeYOLO)

    shelf_detector = detector.ShelfDetector()

    assert isinstance(shelf_detector.model, FakeYOLO)
    assert shelf_detector.model.model_path == MODEL_PATH


def test_detect_returns_valid_detection_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detector output should match the project detection contract."""
    monkeypatch.setattr(detector, "YOLO", FakeYOLO)
    shelf_detector = detector.ShelfDetector()

    detections = shelf_detector.detect("sample.jpg")

    assert isinstance(detections, list)
    assert detections

    for item in detections:
        assert isinstance(item["class_name"], str)
        assert isinstance(item["confidence"], float)
        assert isinstance(item["bbox"], list)
        assert len(item["bbox"]) == 4
        assert all(isinstance(value, (int, float)) for value in item["bbox"])


def test_detect_handles_empty_image_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detector should return an empty detection list when no objects are found."""
    monkeypatch.setattr(detector, "YOLO", FakeYOLO)
    shelf_detector = detector.ShelfDetector()

    detections = shelf_detector.detect("empty.jpg")

    assert detections == []


def test_detect_raises_on_invalid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid source input should fail fast and surface the model error."""
    monkeypatch.setattr(detector, "YOLO", FakeYOLO)
    shelf_detector = detector.ShelfDetector()

    with pytest.raises(ValueError, match="Invalid input source"):
        shelf_detector.detect(None)