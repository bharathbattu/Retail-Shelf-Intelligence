"""End-to-end tests for image-to-insights shelf intelligence pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from analytics import analyze_detections, detect_gaps, evaluate_stock


TEST_DATA_FILE = Path(__file__).resolve().parent.parent / "test_data" / "sample_detections.json"


def load_test_data() -> dict[str, list[dict[str, object]]]:
    """Load shared JSON fixtures used by end-to-end tests."""
    with TEST_DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


class FakeDetector:
    """Mock detector used to avoid heavy YOLO inference in pipeline tests."""

    def __init__(self, detections: list[dict[str, object]]) -> None:
        self._detections = detections

    def detect(self, source: str, confidence_threshold: float | None = None) -> list[dict[str, object]]:
        if not source:
            raise ValueError("Source path is required")

        del confidence_threshold
        return self._detections

    def get_annotated_frame(self) -> np.ndarray:
        return np.zeros((32, 32, 3), dtype=np.uint8)


def test_e2e_pipeline_flow_is_consistent() -> None:
    """Validate detect -> analyze -> evaluate_stock -> detect_gaps full workflow."""
    test_data = load_test_data()
    fake_detector = FakeDetector(test_data["multiple_objects"])

    detections = fake_detector.detect("mock-image.jpg", confidence_threshold=0.45)
    analysis = analyze_detections(detections)
    alerts = evaluate_stock(analysis)
    gap_count = detect_gaps(detections)

    assert analysis["total_items"] == len(detections)
    assert analysis["category_counts"]["bottle"] == 2
    assert analysis["category_counts"]["cup"] == 1
    assert isinstance(alerts, list)
    assert gap_count == 2


def test_e2e_pipeline_builds_ui_ready_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate app-level processing returns a complete dashboard payload."""
    pytest.importorskip("streamlit")
    import app

    test_data = load_test_data()
    detections = test_data["multiple_objects"] + [test_data["edge_cases"][0]]
    fake_detector = FakeDetector(detections)

    monkeypatch.setattr(app, "load_detector", lambda: fake_detector)

    payload = app.process_uploaded_image(
        file_bytes=b"fake-image-bytes",
        filename="shelf.jpg",
        confidence_threshold=0.4,
    )

    assert payload["analysis"]["total_items"] == len(detections)
    assert payload["gap_count"] == detect_gaps(detections)
    assert isinstance(payload["alerts"], list)
    assert payload["primary_category"].startswith("Bottle")
    assert list(payload["category_df"].columns) == ["Category", "Count", "Share"]


def test_pipeline_batch_inputs_do_not_crash() -> None:
    """Batch pipeline simulation should handle varied detection payloads safely."""
    test_data = load_test_data()
    payloads = [
        test_data["multiple_objects"],
        test_data["edge_cases"],
        test_data["empty_detections"],
    ]

    for detections in payloads:
        analysis = analyze_detections(detections)
        alerts = evaluate_stock(analysis)
        gap_count = detect_gaps(detections)

        assert analysis["total_items"] == len(detections)
        assert isinstance(alerts, list)
        assert isinstance(gap_count, int)
        assert gap_count >= 0


def test_pipeline_with_mocks_meets_basic_performance_budget() -> None:
    """Mocked end-to-end execution should stay under a lightweight time budget."""
    test_data = load_test_data()
    start = time.perf_counter()

    detections = test_data["multiple_objects"]
    analysis = analyze_detections(detections)
    _alerts = evaluate_stock(analysis)
    _gaps = detect_gaps(detections)

    elapsed_seconds = time.perf_counter() - start
    assert elapsed_seconds < 1.0