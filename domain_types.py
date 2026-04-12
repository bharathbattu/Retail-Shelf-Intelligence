"""Shared type definitions for detection and analytics data structures."""

from typing import TypedDict


class Detection(TypedDict):
    """Single object detection output from the model."""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class Analysis(TypedDict):
    """Summary analytics generated from a list of detections."""

    total_items: int
    category_counts: dict[str, int]
