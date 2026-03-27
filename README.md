# Traffic_flow_detection

This project runs YOLOv8 + DeepSORT for multi-lane vehicle detection and counting.

## Accuracy Tuning

The tracker pipeline is now tuned for better capture quality:

- Correct DeepSORT input format (`tlwh`) to remove oversized/unstable boxes.
- Higher default image size (`640`) for better small-vehicle recall.
- Detection filtering by min/max box area to suppress noise.
- Class-aware confidence thresholds to keep trucks/buses from being missed.
- More stable tracking defaults to reduce ID switches.

## Runtime Tuning

Set these environment variables before starting the API:

- `MODEL_WEIGHTS` (default: `yolov8s.pt`)
- `VIDEO_SOURCE` (default: `test_video2.mp4`)
- `DETECTION_CONF` (default: `0.30`)
- `DETECTION_IOU` (default: `0.50`)
- `DETECTION_IMGSZ` (default: `640`)
- `MIN_BOX_AREA` (default: `900`)
- `MAX_BOX_AREA_RATIO` (default: `0.45`)
- `TRACK_MAX_AGE` (default: `25`)
- `TRACK_N_INIT` (default: `2`)
- `TRACK_MAX_IOU_DISTANCE` (default: `0.70`)
- `TRACK_MAX_COSINE_DISTANCE` (default: `0.20`)

## Quick Presets

- High recall (capture more vehicles):
  - `DETECTION_CONF=0.24`
  - `DETECTION_IOU=0.55`
  - `TRACK_MAX_AGE=30`
- High precision (fewer false positives):
  - `DETECTION_CONF=0.36`
  - `DETECTION_IOU=0.45`
  - `MIN_BOX_AREA=1200`
