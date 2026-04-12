"""Unit tests for analytics summary generation."""

from __future__ import annotations

import json
from pathlib import Path

from analytics import analyze_detections


TEST_DATA_FILE = Path(__file__).resolve().parent.parent / "test_data" / "sample_detections.json"


def load_test_data() -> dict[str, list[dict[str, object]]]:
    """Load reusable detection fixtures for analytics and pipeline tests."""
    with TEST_DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_analyze_detections_returns_expected_totals_and_categories() -> None:
    """Analytics should compute total count and category breakdown accurately."""
    test_data = load_test_data()
    detections = test_data["multiple_objects"] + test_data["edge_cases"]

    analysis = analyze_detections(detections)

    assert analysis["total_items"] == len(detections)
    assert analysis["category_counts"] == {
        "bottle": 2,
        "box": 1,
        "cup": 1,
        "person": 1,
    }


def test_analyze_detections_handles_empty_input() -> None:
    """Analytics should return a zeroed summary when no detections are provided."""
    test_data = load_test_data()

    analysis = analyze_detections(test_data["empty_detections"])

    assert analysis["total_items"] == 0
    assert analysis["category_counts"] == {}