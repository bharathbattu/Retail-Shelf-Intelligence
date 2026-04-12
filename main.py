"""Entry point for running shelf detection on a single image."""

from analytics import analyze_detections, detect_gaps, evaluate_stock
from config import IMAGE_PATH
from detector import ShelfDetector
from domain_types import Analysis
from utils import validate_image_path


def print_analysis_report(analysis: Analysis) -> None:
    """Print a clean, human-readable shelf analysis report."""
    print("\U0001F4CA Shelf Analysis")
    print("------------------")
    print(f"Total Items: {analysis['total_items']}")
    print()
    print("Category Breakdown:")

    category_counts: dict[str, int] = analysis["category_counts"]
    if not category_counts:
        print("- No objects detected")
        return

    # Print each category and its item count on a separate line.
    for category_name, count in category_counts.items():
        print(f"- {category_name}: {count}")


def print_stock_alerts(alerts: list[str]) -> None:
    """Print stock alerts based on category threshold checks."""
    print()
    print("Stock Alerts:")

    if not alerts:
        print("\u2705 Stock levels are sufficient")
        return

    # Print one alert per line to keep the report easy to scan.
    for alert in alerts:
        print(f"- {alert}")


def print_gap_report(gap_count: int) -> None:
    """Print the shelf gap summary."""
    print()
    print("Shelf Gaps:")

    if gap_count == 0:
        print("\u2705 No significant gaps detected")
        return

    print(f"- Gaps detected: {gap_count}")


def main() -> None:
    """Load the model, run detection, and print a shelf analysis report."""
    try:
        image_path = validate_image_path(IMAGE_PATH)
        detector = ShelfDetector()
        detections = detector.detect(str(image_path))
    except (FileNotFoundError, ImportError) as error:
        # Keep the entry point friendly for early project setup.
        print(f"Setup error: {error}")
        return

    # Turn raw detector output into a simple analytics summary.
    analysis = analyze_detections(detections)
    alerts = evaluate_stock(analysis)
    gap_count = detect_gaps(detections)

    print_analysis_report(analysis)
    print_stock_alerts(alerts)
    print_gap_report(gap_count)

    # Optional video usage:
    # from video_processor import VideoProcessor
    # video_processor = VideoProcessor(detector, frame_skip=2)
    # video_processor.process_video("videos/shelf.mp4")


if __name__ == "__main__":
    main()
