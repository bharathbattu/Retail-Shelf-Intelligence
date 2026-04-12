"""Model loading and image detection logic."""

from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local env
    cv2 = None

try:
    from ultralytics import YOLO
except Exception as import_error:  # pragma: no cover - depends on local env
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = import_error
else:
    ULTRALYTICS_IMPORT_ERROR = None

from config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MODEL_PATH
from domain_types import Detection


class ShelfDetector:
    """Wrapper around a YOLOv8 model for shelf image detection."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        if YOLO is None:
            raise RuntimeError(
                f"Ultralytics failed to load due to: {ULTRALYTICS_IMPORT_ERROR}"
            ) from ULTRALYTICS_IMPORT_ERROR

        # Load once on startup. Ultralytics downloads the model automatically if needed.
        try:
            self.model = YOLO(model_path)
            # Streamlit Cloud runs on CPU-only infrastructure.
            self.model.to("cpu")
        except Exception as error:
            raise RuntimeError(f"Model initialization failed: {error}") from error
        self.last_result: Any | None = None

    def detect(
        self,
        source: str | Any,
        confidence_threshold: float | None = None,
    ) -> list[Detection]:
        """Run detection on an image path or frame and return formatted detections."""
        confidence = CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold

        results = self.model.predict(
            source=source,
            conf=max(0.0, min(float(confidence), 1.0)),
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        return self._format_results(results)

    def get_annotated_frame(self) -> Any | None:
        """Return the most recent YOLO-annotated frame, if available."""
        if self.last_result is None:
            return None

        return self._render_minimal_annotations(self.last_result)

    def _render_minimal_annotations(self, result: Any) -> Any:
        """Draw cleaner overlays with thin boxes and class-only labels."""
        # Improvement 1: reduce clutter by removing confidence text and shrinking overlays.
        if cv2 is None:
            return result.plot(conf=False, line_width=1, font_size=10)

        original_frame = getattr(result, "orig_img", None)
        if original_frame is None:
            return result.plot(conf=False, line_width=1, font_size=10)

        frame = original_frame.copy()
        frame_height, frame_width = frame.shape[:2]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        box_thickness = 1
        text_thickness = 1

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(result.names[class_id])
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]

            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            # Use a stable, readable color to keep annotations visible without being heavy.
            color = (16, 148, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            text_size, baseline = cv2.getTextSize(
                class_name,
                font_face,
                font_scale,
                text_thickness,
            )
            text_width, text_height = text_size

            label_left = x1
            label_top = max(0, y1 - text_height - baseline - 6)
            label_bottom = label_top + text_height + baseline + 4
            label_right = min(frame_width - 1, label_left + text_width + 8)

            cv2.rectangle(
                frame,
                (label_left, label_top),
                (label_right, label_bottom),
                color,
                thickness=-1,
            )
            cv2.putText(
                frame,
                class_name,
                (label_left + 4, label_bottom - baseline - 2),
                font_face,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        return frame

    def _format_results(self, results: list[Any]) -> list[Detection]:
        """Convert YOLO results into a simple Python list of dictionaries."""
        if not results:
            self.last_result = None
            return []

        result = results[0]
        self.last_result = result
        names = result.names
        detections: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_name = str(names[class_id])

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

        return detections
