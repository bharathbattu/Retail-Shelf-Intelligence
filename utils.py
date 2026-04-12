"""Utility helpers used across the project."""

from pathlib import Path

from domain_types import Detection


def validate_image_path(image_path: str | Path) -> Path:
    """Return a resolved image path and ensure the file exists."""
    resolved_path = Path(image_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Image not found: {resolved_path}")

    return resolved_path


def print_detection_summary(detections: list[Detection]) -> None:
    """Print a simple summary of the detection results."""
    print(f"Detected objects: {len(detections)}")
