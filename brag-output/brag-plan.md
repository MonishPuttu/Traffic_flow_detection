# /brag plan — Traffic_flow_detection

**What it is:** Multi-lane vehicle detection and counting — YOLOv8 detects, DeepSORT tracks, and a track crossing a lane line is counted — served by FastAPI with a live `/video_feed` viewer and Prometheus metrics.
**Who it's for:** Traffic/smart-city engineers and anyone evaluating CV + MLOps skills.
**What sets it apart:** It's not just a notebook — tuned detection filters, stable tracking, a live dashboard and scrape-able telemetry, all Dockerized.
**Most impressive claim:** 31.3 FPS with 12 ms inference and 0 ID switches in the live dashboard.
**Visual hook:** Real annotated footage with boxes, IDs and lane lines — "Every vehicle. Detected, tracked, counted."
**Tone:** polished — the dashboard's own light palette and Manrope.
**Share caption:** "Traffic flow, measured."

## What's real
- **Footage + counts:** `traffic_analysis.run_traffic_analysis` from this repo, unmodified, was run (CPU) on a public 38-second street clip (`input_video.mp4` from github.com/ahmetozlu/vehicle_counting_tensorflow, upscaled to 1280×704) with `yolov8s.pt`, two count lines placed for that street via `COUNT_LINES`, and `frame_callback` recording the annotated frames. Result: car 8, truck 1, 4,053 detections, 1 ID switch, 906 frames. The on-screen counters are that run's per-frame `metrics_dict`. (The repo's own `test_video2.mp4` is not committed, so it couldn't be used.)
- **Dashboard:** the viewer screenshot (31.3 FPS, 12.06 ms inference, 15.82 ms tracking, 0 ID switches) is the project's own capture used on the Portfolio site (`thumbnails/trafficflow.webp`).
- **Prometheus lines:** metric names from `api_server.py`, values from the run above.
- Colors/font from the `/viewer` HTML in `api_server.py`: `--bg #f5f7fb`, `--good #2a7b61`, Manrope.

## Storyboard (21s, 1920×1080 @ 30fps)
| # | Time | Scene | On screen |
|---|------|-------|-----------|
| 1 | 0.0–3.4 | **Hook** | Real annotated footage, live count chip — "Every vehicle. Detected, tracked, counted." |
| 2 | 3.4–6.6 | **Reveal** | "Traffic Flow Monitoring" + "Live roadway analysis with low-latency performance telemetry" + stack pills |
| 3 | 6.6–11.2 | **How it counts** | Footage in the viewer card with live car/truck/track counters; 1 YOLOv8 detects · 2 DeepSORT tracks · 3 Lines count |
| 4 | 11.2–15.4 | **Telemetry** | Real dashboard screenshot + "31.3 FPS" callout |
| 5 | 15.4–18.4 | **Observability** | `curl localhost:8000/prometheus` output |
| 6 | 18.4–21.0 | **Outro** | "Traffic flow, measured." + GitHub link |

## Voice-over version (43s)
The final `brag.mp4` is the extended cut with narration. Voice: Kokoro TTS (`af_bella`), generated locally. Each scene's timeline was stretched to fit its line (entrances and transitions keep their original speed; only the hold in the middle of each scene slows down), the soundtrack was re-timed to match, and the music ducks under the voice. Some spellings below are written for the voice, e.g. "Ani-Talk", "R-x Check".

| # | Time | Narration |
|---|------|-----------|
| 1 | 0.0–4.3s | Every vehicle on the road. Detected, tracked, and counted. |
| 2 | 4.3–10.9s | This is Traffic Flow Monitoring: live roadway analysis, with low-latency telemetry. |
| 3 | 10.9–23.0s | YOLO v8 detects cars, trucks, buses and motorcycles. Deep Sort gives each one a stable ID across frames. And when a track crosses a lane line, it's counted. |
| 4 | 23.0–33.2s | A live dashboard streams the annotated video, with performance metrics. Here, thirty-one frames per second, with twelve millisecond inference. |
| 5 | 33.2–38.7s | Every metric is exposed to Prometheus, so you can graph it, and alert on it. |
| 6 | 38.7–43.4s | Traffic flow, measured. Find it on GitHub. |
