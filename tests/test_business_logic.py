"""Unit tests for stock-alert and gap-detection business logic."""

from __future__ import annotations

import json
from pathlib import Path

from analytics import detect_gaps, evaluate_stock


TEST_DATA_FILE = Path(__file__).resolve().parent.parent / "test_data" / "sample_detections.json"


def load_test_data() -> dict[str, list[dict[str, object]]]:
    """Load reusable detection fixtures for business rule tests."""
    with TEST_DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_evaluate_stock_triggers_alert_when_below_threshold() -> None:
    """A category below configured threshold should produce a low-stock alert."""
    analysis = {
        "total_items": 2,
        "category_counts": {"bottle": 2},
    }

    alerts = evaluate_stock(analysis)

    assert len(alerts) == 1
    assert "Low stock: bottle" in alerts[0]


def test_evaluate_stock_returns_no_alerts_when_threshold_is_met() -> None:
    """A category at or above threshold should not generate alert messages."""
    analysis = {
        "total_items": 6,
        "category_counts": {"bottle": 6},
    }

    alerts = evaluate_stock(analysis)

    assert alerts == []


def test_detect_gaps_counts_significant_horizontal_gaps() -> None:
    """Gap detection should count only gaps larger than configured threshold."""
    detections = load_test_data()["multiple_objects"]

    gap_count = detect_gaps(detections)

    assert gap_count == 2


def test_detect_gaps_ignores_invalid_bbox_entries() -> None:
    """Invalid bounding-box shapes should not break gap detection."""
    detections = [
        {
            "class_id": 0,
            "class_name": "bottle",
            "confidence": 0.9,
            "bbox": [0.0, 0.0, 20.0, 100.0],
        },
        {
            "class_id": 1,
            "class_name": "cup",
            "confidence": 0.8,
            "bbox": [25.0, 0.0, 40.0],
        },
    ]

    gap_count = detect_gaps(detections)

    assert gap_count == 0