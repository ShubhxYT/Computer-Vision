# Computer Vision Toolkit

![Python](https://img.shields.io/badge/Python-3.10-blue) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green) ![YOLOv8](https://img.shields.io/badge/Ultralytics-8.2.82-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-red)

> Six self-contained computer vision modules built following a 12-hour tutorial — everything from pose estimation and face mesh to YOLO object counting and custom license plate OCR. My first GitHub push.

## What I Built It For

I'd never touched computer vision before August 2024. Didn't know what CUDA was, had never pushed to GitHub, Python experience was basic scripting. I found a 12-hour CV tutorial and worked through it over a few days — pausing, rewinding, typing every line. The license plate recognition module went further: I trained my own YOLOv8 model on Indian number plates and built the full detect → track → OCR pipeline from scratch. One weekend in November I pushed everything — my first four repos, my first GitHub portfolio. This is where I started.

## Modules

Six independent modules — no shared code between them. Each has its own entry point, models, and processing pipeline.

### 1. Pose Estimation (`pose_tracking/`)
Real-time 33-point body landmark detection using MediaPipe BlazePose. Wraps the raw MediaPipe API in a reusable `poseDetector` class with find/draw/extract methods. The demo highlights left and right elbows with FPS overlay.

- **Model:** MediaPipe Pose (BlazePose)
- **Entry points:** `main.py` (class-based demo), `pose_tracking.py` (raw API demo)
- **Output:** Annotated video with skeleton, landmark IDs, and FPS counter

### 2. Face Detection & Mesh (`FaceDetection/`)
Two approaches to face analysis: bounding box detection (6 landmarks) and dense 468-point face mesh. Includes a combined script running face detection, face mesh, hand tracking, and pose estimation simultaneously — all four MediaPipe models in one loop.

- **Model:** MediaPipe Face Detection + Face Mesh
- **Scripts:** `facedetection.py` / `facedetection_module.py` (detection), `facemesh.py` / `facemesh_module.py` (mesh), `mix.py` (all-in-one)
- **Output:** Bounding boxes with confidence %, 468-point wireframe mesh, landmark IDs

### 3. Hand Tracking & Gesture Control (`Hand_Tracking/`)
21-point hand landmark detection with a full `handDetector` class exposing find, position extraction, finger state, and distance measurement. Three applications built on top: finger counting with visual overlays, gesture-based volume control (thumb-index pinch mapped to Windows system volume via PyCAW), and a virtual paint canvas with color selection and eraser.

- **Model:** MediaPipe Hands
- **Applications:** Finger counter (with middle-finger detection), gesture volume controller, virtual paint
- **Hardware:** Webcam + Windows audio (PyCAW)

### 4. Object Detection & Counting (`Object_Detection/`)
YOLOv8 across three scales: nano (basics), medium (webcam), large (counting). Real-time webcam detection of 80 COCO classes with corner bounding boxes and confidence labels. Two specialized counters: people counter with dual-line direction-aware counting and region masking, vehicle counter with single-line highway zone isolation.

- **Model:** YOLOv8n / YOLOv8m / YOLOv8l
- **Tracker:** SORT (Kalman filter + Hungarian algorithm)
- **Scripts:** `yolo_basics.py`, `yolo_webcam.py`, `people_counter.py`, `car_counter.py`, `training-ppt.py`
- **Features:** Region masking, line-cross detection with ID deduplication, graphics overlays

### 5. License Plate Recognition (`Plate_recon/`)
The most sophisticated module. Dual YOLO pipeline: one model detects vehicles (COCO-pretrained), another detects license plates (custom-trained on Indian plates). Both feed into independent SORT trackers. Detected plates are cropped, thresholded with OTSU, and passed through EasyOCR. Results go through Indian format validation (XX NN XXX) with character-level correction for common OCR mistakes. Spatial association matches plates to vehicles via bounding box containment. Post-processing interpolates missed frames and re-renders video with overlays.

- **Models:** YOLOv8n (cars) + custom `plate_recon.pt` (plates)
- **OCR:** EasyOCR with Indian license format validation and correction
- **Pipeline:** Detect → Track → Crop → Preprocess → OCR → Validate → Associate → Export CSV → Visualize
- **Scripts:** `main.py` (real-time), `fail.py` (structured pipeline with CSV export), `visualize.py` (offline re-rendering), `add_missing_data.py` (bounding box interpolation)
- **SORT tracker:** Bundled full implementation with 11 MOT benchmark sequences

### 6. Document Scanner (`File_Scanner/`)
Real-time document scanning from webcam feed. Finds the largest rectangular contour, applies perspective warp to flatten it, then runs adaptive thresholding for a clean scanned output. Interactive Canny threshold tuning via trackbars. Press 's' to save scanned image.

- **Pipeline:** Grayscale → Gaussian blur → Canny edge → Dilation + Erosion → Contour detection → Largest 4-point contour → Perspective warp → Adaptive threshold
- **Features:** Interactive trackbar tuning, save-to-disk

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Computer-Vision/                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Pose         │  │ Face         │  │ Hand          │  │
│  │ Estimation   │  │ Detection    │  │ Tracking      │  │
│  │ (MediaPipe)  │  │ (MediaPipe)  │  │ (MediaPipe)   │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Object       │  │ License      │  │ Document      │  │
│  │ Detection    │  │ Plate OCR    │  │ Scanner       │  │
│  │ (YOLOv8)     │  │ (YOLO+OCR)   │  │ (OpenCV)      │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                         │
│  All modules are independent — no shared code or imports │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| Computer Vision | OpenCV 4.10, cvzone 1.5.6, scikit-image 0.19.3 |
| Deep Learning | PyTorch 2.4.0+cu118, TorchVision 0.19 |
| Object Detection | Ultralytics YOLOv8 8.2.82 (nano, medium, large) |
| Pose & Face & Hands | MediaPipe 0.10.14 |
| OCR | EasyOCR 1.7.0 |
| Object Tracking | SORT (Kalman Filter + Hungarian), filterpy 1.4.5 |
| Numerical | NumPy 1.26.4, SciPy 1.10.1, Pandas 2.2.2 |
| Visualization | Matplotlib 3.9.2, Seaborn 0.13.2 |
| Environment | pip (requirements.txt), Conda (environment.yml) |

## Setup & Usage

### Prerequisites
- Python 3.10+
- CUDA-capable GPU recommended (CPU fallback works but slower)
- Webcam (for real-time modules)

### Installation
```bash
git clone https://github.com/ShubhxYT/Computer-Vision.git
cd Computer-Vision

# Option 1: pip
pip install -r requirements.txt

# Option 2: conda
conda env create -f environment.yml
conda activate cv
```

### Running Each Module

#### Pose Estimation
```bash
cd pose_tracking
# Update the video path in main.py first (currently hardcoded to D:/Codes/...)
python main.py          # Class-based demo with elbow highlighting
python pose_tracking.py  # Raw MediaPipe API demo
```

#### Face Detection & Mesh
```bash
cd FaceDetection
python facedetection_module.py   # Face detection with corner bounding boxes
python facemesh_module.py        # 468-point face mesh
python mix.py                    # All MediaPipe models running together
```

#### Hand Tracking
```bash
cd Hand_Tracking
python HandTrackingModule.py     # Hand detection with distance measurement
python finger-count.py           # Finger counting with overlay graphics
python finger-count-middle.py    # Includes middle-finger detection
python gesture-volume-controller.py  # Pinch to control system volume
python virtual-paint/virtual_paint.py  # Draw on screen with finger gestures
```

#### Object Detection
```bash
cd Object_Detection
python yolo_webcam.py       # Real-time webcam detection (80 COCO classes)
python people_counter.py    # People counting with line-cross detection
python car_counter.py       # Vehicle counting on highway footage
python training-ppt.py      # YOLO fine-tuning template for PPE detection
```

#### License Plate Recognition
```bash
cd Plate_recon
python main.py              # Real-time dual YOLO + SORT pipeline
python fail.py              # Structured pipeline with CSV export
python add_missing_data.py  # Interpolate missed bounding boxes
python visualize.py         # Offline re-rendering with plate overlays
```

#### Document Scanner
```bash
cd File_Scanner
python main.py              # Real-time document scanning from webcam
```

> **Note:** All video paths are hardcoded to `D:/Codes/...` (Windows development environment). Update them to your local paths before running. YOLO weights download automatically on first run via Ultralytics. The custom `plate_recon.pt` model for license plate detection must be obtained separately.

## Project Structure

```
Computer-Vision/
├── pose_tracking/
│   ├── main.py                  # Entry: elbow-tracking demo
│   ├── pose_module_tracking.py  # poseDetector class (MediaPipe wrapper)
│   ├── pose_tracking.py         # Raw MediaPipe pose script
│   ├── image.png                # Test image
│   └── videos/                  # 1.mp4 - 4.mp4
├── FaceDetection/
│   ├── facedetection.py         # Raw face detection script
│   ├── facedetection_module.py  # FaceDetector class wrapper
│   ├── facemesh.py              # Raw face mesh script
│   ├── facemesh_module.py       # FaceMeshDetector class wrapper
│   └── mix.py                   # All MediaPipe models combined
├── Hand_Tracking/
│   ├── HandTracking.py          # Raw hand detection
│   ├── HandTrackingModule.py    # handDetector class (find, position, fingersUp, findDistance)
│   ├── finger-count.py          # Finger counting app
│   ├── finger-count-middle.py   # Finger counting + middle finger
│   ├── gesture-volume-controller.py  # Pinch-to-volume
│   └── virtual-paint/
│       └── virtual_paint.py     # Finger painting canvas
├── Object_Detection/
│   ├── yolo_basics.py           # Minimal YOLO inference (5 lines)
│   ├── yolo_webcam.py           # Real-time webcam detection
│   ├── people_counter.py        # People counter with SORT tracking
│   ├── car_counter.py           # Vehicle counter with region masking
│   ├── training-ppt.py          # YOLO fine-tuning template
│   ├── sort.py                  # SORT tracker (duplicate)
│   ├── images/                  # Mask PNGs, graphics overlays
│   └── videos/                  # Test videos (people, cars, bikes, motorbikes, PPE)
├── Plate_recon/
│   ├── main.py                  # Real-time dual YOLO + SORT pipeline
│   ├── fail.py                  # Structured pipeline with CSV export
│   ├── util.py                  # OCR, format validation, character correction, CSV writer
│   ├── visualize.py             # Offline re-rendering with overlays
│   ├── add_missing_data.py      # Bounding box interpolation
│   ├── licence_plate.mp4        # Input video
│   ├── requirements.txt         # Module-specific dependencies
│   └── sort/                    # SORT tracker (full implementation + MOT benchmarks)
├── File_Scanner/
│   ├── main.py                  # Real-time document scanner
│   └── utlis.py                 # Stack images, contour processing, trackbars
├── requirements.txt             # Full project dependencies (pinned)
└── environment.yml              # Conda environment specification
```

## Key Design Decisions

- **Six separate modules instead of a unified toolkit** — intentional, mirrors the tutorial structure. Each module is a self-contained learning artifact.
- **Class-based wrappers for MediaPipe** — `poseDetector`, `handDetector`, `FaceDetector`, `FaceMeshDetector` encapsulate the boilerplate (BGR→RGB conversion, landmark extraction, drawing) into clean, reusable APIs.
- **Dual YOLO + dual SORT in license plate module** — two independent detection+tracking pipelines running simultaneously (vehicles + plates), with spatial association via bounding box containment.
- **Position-aware OCR correction** — Indian license plates follow XX NN XXX format. The correction maps use different dictionaries for letter positions (0,1,4,5,6) vs digit positions (2,3), fixing common OCR mistakes like O↔0, I↔1, S↔5.
- **Line-cross deduplication** — Both counters track which IDs have already been counted to prevent double-counting when an object straddles the counting line across multiple frames.
- **Region masking with PNG overlays** — Instead of complex ROI configuration, both counters use mask images with `cv2.bitwise_and` to isolate detection zones, making zone changes as simple as editing a PNG.
- **Frame interpolation for tracking gaps** — `add_missing_data.py` uses scipy's `interp1d` to linearly interpolate bounding boxes in frames where YOLO missed detections, improving continuity for offline analysis.

## Screenshots

Screenshots unavailable — project requires GPU, YOLO model downloads, and webcam/video files with hardcoded Windows paths that don't resolve in CI/demo environments. Each module produces real-time annotated video output with bounding boxes, landmark skeletons, FPS counters, and tracking overlays.
