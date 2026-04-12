"""Project configuration for the Retail Shelf Intelligence System."""

from pathlib import Path


# Base directory for the project.
BASE_DIR = Path(__file__).resolve().parent

# YOLOv8 model configuration.
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Input image configuration.
IMAGE_PATH = BASE_DIR / "images" / "shelf.jpg"

# Default minimum item count used when a category does not have its own rule.
DEFAULT_THRESHOLD = 5

# Category-specific minimum stock thresholds.
STOCK_THRESHOLDS = {
    "bottle": 5,
    "cup": 5,
    "person": 1,
}

# Minimum horizontal distance, in pixels, to count as a shelf gap.
GAP_THRESHOLD = 50
