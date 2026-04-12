# Retail Shelf Intelligence System

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111111?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

An AI-powered computer vision project for monitoring retail shelves from images and video streams. The system uses YOLOv8 for object detection, OpenCV for video processing, and Streamlit for an interactive analytics dashboard to help identify product availability, low-stock risks, and visible shelf gaps.

It is designed as a lightweight, practical foundation for retail automation use cases such as shelf auditing, stock visibility checks, and category-level merchandising insights.

## Key Features

- Detects products and shelf objects from retail shelf images
- Supports both image analysis and video-frame processing pipelines
- Counts detected items automatically
- Generates category-wise analytics from detection results
- Flags low-stock conditions using configurable thresholds
- Detects horizontal shelf gaps that may indicate empty display space
- Provides a Streamlit dashboard with clean, class-only annotations for better readability
- Includes processing status feedback and a reset flow for faster image re-analysis
- Uses a modular Python codebase that is easy to extend for custom retail classes

## Tech Stack

| Layer | Technology |
| --- | --- |
| Programming Language | Python |
| Detection Model | YOLOv8 (`ultralytics`) |
| Computer Vision | OpenCV |
| Dashboard | Streamlit |
| Data Handling | Native Python structures, TypedDict-based domain models |
| Inference Asset | Auto-downloaded `yolov8n.pt` via Ultralytics |

## System Architecture

The project follows a simple pipeline-based design:

1. An input image or video frame is passed into the detection layer.
2. `ShelfDetector` loads the YOLOv8 model and returns structured detections.
3. `analytics.py` converts raw detections into business-friendly insights such as total item count, category breakdown, low-stock alerts, and shelf-gap estimates.
4. Results are shown either through:
   - `main.py` for CLI-based analysis
   - `app.py` for the Streamlit dashboard
   - `video_processor.py` for continuous frame-by-frame video analysis

### Flow Overview

```text
Image / Video
     |
     v
YOLOv8 Detection Engine (`detector.py`)
     |
     v
Analytics Layer (`analytics.py`)
     |
     +--> CLI Report (`main.py`)
     +--> Streamlit Dashboard (`app.py`)
     +--> Video Overlay Output (`video_processor.py`)
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/retail-shelf-intelligence-system.git
cd retail-shelf-intelligence-system
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Project Assets

Make sure this local file is present before running the app:

- `images/shelf.jpg` for the default CLI image example

The YOLOv8 model (`yolov8n.pt`) is automatically downloaded at runtime.

## How to Run

### Run the CLI Image Analysis

The CLI entry point analyzes the default shelf image configured in `config.py`.

```bash
python main.py
```

This will:

- load the YOLOv8 model
- run detection on the configured image
- print total detected items
- show category-wise counts
- report low-stock alerts
- report detected shelf gaps

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## Deployment

This app is deployed using Streamlit Community Cloud.

Run locally:

```bash
streamlit run app.py
```

The YOLOv8 model is automatically downloaded at runtime.

## Deployment Notes

- This app runs in a headless cloud environment
- Uses opencv-python-headless to avoid GUI dependencies
- Avoids libGL-related errors
- YOLOv8 model downloads automatically at runtime

## Deployment Fix

This project uses:

- opencv-python-headless for compatibility
- packages.txt to install libGL system dependency

This prevents:

- libGL.so.1 errors in Streamlit Cloud

The dashboard currently supports uploaded shelf images and displays:

- annotated detection image with minimal class-only labels
- total item count
- unique category count
- shelf gap count
- category breakdown chart and table
- low-stock alerts
- processing loader and completion confirmation
- reset analysis button to clear state and upload a new image

## Testing

This repository uses `pytest` for end-to-end and module-level validation.

Run all tests:

```bash
pytest tests/
```

### Run Video Analysis from Python

The video pipeline is available through `video_processor.py`. A minimal example:

```bash
python -c "from detector import ShelfDetector; from video_processor import VideoProcessor; VideoProcessor(ShelfDetector(), frame_skip=2).process_video('path/to/video.mp4')"
```

The video pipeline runs headlessly and does not open GUI windows.

## Example Outputs

### CLI Report

```text
Shelf Analysis
------------------
Total Items: <detected_count>

Category Breakdown:
- <category_1>: <count>
- <category_2>: <count>

Stock Alerts:
- Low stock: <category_name> (<count> items, threshold <threshold>)

Shelf Gaps:
- Gaps detected: <gap_count>
```

### Streamlit Dashboard

Typical dashboard output includes:

- an annotated shelf image with clean bounding boxes and class-only labels
- top-level KPI cards for total items, unique categories, and shelf gaps
- a category count chart and table
- stock sufficiency warnings

### Sample Input Asset

The repository already includes a sample shelf image at `images/shelf.jpg`.

## Folder Structure

```text
Retail Shelf Intelligence System/
|-- app.py                # Streamlit dashboard
|-- main.py               # CLI entry point for image analysis
|-- detector.py           # YOLOv8 model wrapper and detection formatting
|-- analytics.py          # Counting, stock alerts, and gap detection
|-- video_processor.py    # OpenCV-based video processing pipeline
|-- config.py             # Model paths, thresholds, and runtime settings
|-- domain_types.py       # Shared typed data structures
|-- utils.py              # Utility helpers
|-- requirements.txt      # Runtime dependencies for local/dev/cloud
`-- images/
    |-- shelf.jpg         # Sample shelf image
    `-- .gitkeep
```

## Configuration Notes

Runtime behavior can be adjusted from `config.py`:

- `CONFIDENCE_THRESHOLD` controls detection confidence filtering
- `IOU_THRESHOLD` controls YOLO overlap handling
- `DEFAULT_THRESHOLD` defines the fallback low-stock threshold
- `STOCK_THRESHOLDS` allows category-specific stock rules
- `GAP_THRESHOLD` defines the minimum horizontal gap used for shelf-gap alerts

## Future Improvements

- Train a custom retail dataset instead of relying on generic YOLO classes
- Add support for product SKU recognition and brand-level classification
- Export reports to CSV, Excel, or PDF
- Add real-time webcam and RTSP stream support
- Store historical shelf analytics for trend monitoring
- Build restocking recommendations based on shelf performance
- Add heatmaps and advanced merchandising analytics to the dashboard
- Package the project with Docker for easier deployment

## Resume Description

Built an AI-powered Retail Shelf Intelligence System using Python, YOLOv8, OpenCV, and Streamlit to detect products from shelf images and video, perform product counting, generate category-wise analytics, and identify low-stock conditions and shelf gaps.

Developed a modular computer vision pipeline with a dashboard interface for retail monitoring, combining object detection with business-focused inventory insights.

## Why This Project Stands Out

- Connects raw computer vision output to practical retail operations use cases
- Demonstrates both model inference and product-minded analytics design
- Shows end-to-end engineering across detection, analytics, and UI
- Provides a strong portfolio project for AI, CV, and applied ML roles
