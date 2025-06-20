# Webcam Fire & Smoke Detection

This repository contains a small sample script using OpenCV to detect
potential fire or smoke via your webcam. Detected events are annotated on the
video with a timestamp and also logged to `detection_log.txt`.

## Requirements

Install the required packages with:

```bash
pip install opencv-python numpy
```

## Usage

Run the detector with:

```bash
python fire_smoke_detection.py
```

Press `q` to stop the program.
