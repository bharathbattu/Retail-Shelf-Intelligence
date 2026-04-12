"""Video processing pipeline for retail shelf analysis."""

from collections.abc import Callable
from typing import Any

try:
    import cv2
except ImportError as import_error:  # pragma: no cover - depends on local env
    cv2 = None
    OPENCV_IMPORT_ERROR = import_error
else:
    OPENCV_IMPORT_ERROR = None

from analytics import analyze_detections, detect_gaps, evaluate_stock
from detector import ShelfDetector


class VideoProcessor:
    """Run detection and analytics on video frames using OpenCV."""

    def __init__(self, detector: ShelfDetector, frame_skip: int = 2) -> None:
        if cv2 is None:
            raise ImportError(
                "OpenCV is not installed. Install it with: pip install opencv-python-headless"
            ) from OPENCV_IMPORT_ERROR

        self.detector = detector
        self.frame_skip = max(1, frame_skip)

    def process_video(
        self,
        video_source: str | int,
        on_frame: Callable[[Any], None] | None = None,
    ) -> None:
        """Read a video source and process frames in headless mode.

        Use ``on_frame`` to consume processed frames (for example, to stream,
        save, or forward them) without relying on GUI windows.
        """
        capture = cv2.VideoCapture(video_source)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video source: {video_source}")

        frame_index = 0
        last_output_frame: Any | None = None

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index % self.frame_skip == 0:
                    detections = self.detector.detect(frame)
                    analysis = analyze_detections(detections)
                    alerts = evaluate_stock(analysis)
                    gap_count = detect_gaps(detections)

                    annotated_frame = self.detector.get_annotated_frame()
                    if annotated_frame is None:
                        annotated_frame = frame.copy()

                    output_frame = self._overlay_summary(
                        annotated_frame,
                        analysis["total_items"],
                        len(alerts),
                        gap_count,
                    )
                    last_output_frame = output_frame
                else:
                    # Reuse the last processed frame to keep playback responsive.
                    output_frame = last_output_frame if last_output_frame is not None else frame

                if on_frame is not None:
                    on_frame(output_frame)

                frame_index += 1
        finally:
            capture.release()

    def _overlay_summary(
        self,
        frame: Any,
        total_items: int,
        alert_count: int,
        gap_count: int,
    ) -> Any:
        """Draw a compact analytics summary on the processed frame."""
        overlay_lines = [
            f"Total items: {total_items}",
            f"Low stock alerts: {alert_count}",
            f"Gaps detected: {gap_count}",
        ]

        y_position = 30
        for line in overlay_lines:
            cv2.putText(
                frame,
                line,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y_position += 30

        return frame
