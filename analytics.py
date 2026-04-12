"""Analytics helpers for summarizing shelf detections."""

from collections import Counter

from config import DEFAULT_THRESHOLD, GAP_THRESHOLD, STOCK_THRESHOLDS
from domain_types import Analysis, Detection


def analyze_detections(detections: list[Detection]) -> Analysis:
    """
    Convert raw detection output into a simple analytics summary.

    Returns the total number of detected items and a count per category.
    """
    # Count how many times each detected class appears.
    category_counts = Counter(
        detection["class_name"] for detection in detections if detection["class_name"]
    )

    return {
        "total_items": len(detections),
        "category_counts": dict(sorted(category_counts.items())),
    }


def evaluate_stock(analysis: Analysis) -> list[str]:
    """
    Compare category counts against configured thresholds.

    Returns a list of low-stock alert messages for categories that fall
    below the expected minimum count.
    """
    alerts: list[str] = []
    category_counts = analysis["category_counts"]

    for category_name, count in sorted(category_counts.items()):
        # Prefer a category-specific rule and fall back to the default limit.
        threshold = STOCK_THRESHOLDS.get(category_name, DEFAULT_THRESHOLD)

        if count < threshold:
            alerts.append(
                f"\u26A0\uFE0F Low stock: {category_name} ({count} items, threshold {threshold})"
            )

    return alerts


def detect_gaps(detections: list[Detection]) -> int:
    """
    Count significant horizontal gaps between adjacent detected objects.

    Detections are sorted from left to right using the bounding box x1 value,
    then each neighboring pair is checked for empty horizontal space.
    """
    boxes: list[tuple[float, float, float, float]] = []

    for detection in detections:
        bbox = detection.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            boxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))

    # Compare objects in left-to-right order across the shelf.
    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    gap_count = 0

    for current_box, next_box in zip(sorted_boxes, sorted_boxes[1:]):
        current_x2 = current_box[2]
        next_x1 = next_box[0]
        gap_size = next_x1 - current_x2

        if gap_size > GAP_THRESHOLD:
            gap_count += 1

    return gap_count
