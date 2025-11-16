import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# CONFIG
CONF_THRESH = 0.42
COUNT_LINES = [
    [(165, 238), (397, 243)],
    [(519, 246), (738, 250)],
]
CLASSES_TO_COUNT = ["car", "truck", "bus", "motorcycle"]
MODEL_WEIGHTS = "yolov8s.pt"


# UTILITY FUNCTIONS
def centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def segments_intersect(p1, p2, q1, q2):
    """Return True if line segments p1p2 and q1q2 intersect"""

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


# MAIN TRACKING LOOP
def run_traffic_analysis(
    source="test_video2.mp4", display=True, metrics_dict=None, frame_callback=None
):
    model = YOLO(MODEL_WEIGHTS)

    # Try to use GPU if available
    try:
        model.to("cuda")
        print("✅ Using CUDA GPU")
    except Exception as e:
        print(f"⚠️  Using CPU (slower): {e}")

    tracker = DeepSort(
        max_age=40,
        n_init=3,
        max_iou_distance=0.6,
        max_cosine_distance=0.25,
        nn_budget=100,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    track_memory = {}
    counts = defaultdict(int)
    frame_id = 0

    # Performance tracking variables
    fps_time = time.time()
    inference_times = []
    tracking_times = []
    previous_track_ids = set()
    id_switches = 0
    total_detections = 0
    detection_by_class = defaultdict(int)

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # Inference with timing
        inference_start = time.time()
        results = model.predict(
            frame,
            imgsz=512,
            conf=CONF_THRESH,
            device="cuda" if model.device.type == "cuda" else "cpu",
            verbose=False,
        )
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        if len(inference_times) > 100:
            inference_times.pop(0)

        # Process detections
        dets = []
        frame_detections = 0

        for res in results:
            boxes = res.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if cls_name in CLASSES_TO_COUNT and conf >= CONF_THRESH:
                    dets.append(([x1, y1, x2, y2], conf, cls_name))
                    frame_detections += 1
                    detection_by_class[cls_name] += 1

        total_detections += frame_detections

        # DeepSORT update with timing
        tracking_start = time.time()
        ds_input = [(b, conf, cls) for (b, conf, cls) in dets]
        tracks = tracker.update_tracks(ds_input, frame=frame)
        tracking_time = time.time() - tracking_start
        tracking_times.append(tracking_time)
        if len(tracking_times) > 100:
            tracking_times.pop(0)

        # Track ID switch detection
        current_track_ids = set()

        # Process tracks
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            current_track_ids.add(tid)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            c = centroid((x1, y1, x2, y2))
            cls_name = track.det_class if hasattr(track, "det_class") else "vehicle"

            # Check line crossing for all counting lines
            if tid in track_memory:
                prev_c = track_memory[tid]
                for count_line in COUNT_LINES:
                    if segments_intersect(prev_c, c, count_line[0], count_line[1]):
                        counts[cls_name] += 1

            track_memory[tid] = c
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{cls_name}-{tid}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.circle(frame, c, 3, (255, 0, 0), -1)

        # Detect ID switches (approximate)
        if previous_track_ids:
            disappeared = previous_track_ids - current_track_ids
            new_ids = current_track_ids - previous_track_ids
            id_switches += min(len(disappeared), len(new_ids))
        previous_track_ids = current_track_ids

        # Draw counting lines
        for idx, count_line in enumerate(COUNT_LINES):
            cv2.line(frame, count_line[0], count_line[1], (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Lane {idx + 1}",
                (count_line[0][0], count_line[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Calculate metrics
        current_fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()

        avg_inference_ms = np.mean(inference_times) * 1000 if inference_times else 0
        avg_tracking_ms = np.mean(tracking_times) * 1000 if tracking_times else 0

        # Draw metrics on frame
        cv2.putText(
            frame,
            "Counts: " + str(dict(counts)),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {current_fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Inference: {avg_inference_ms:.1f}ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Active Tracks: {len(current_track_ids)}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

        # Update shared metrics for API and Prometheus
        if metrics_dict is not None:
            metrics_dict["counts"] = dict(counts)
            metrics_dict["fps"] = round(current_fps, 2)
            metrics_dict["inference_time_ms"] = round(avg_inference_ms, 2)
            metrics_dict["tracking_time_ms"] = round(avg_tracking_ms, 2)
            metrics_dict["active_tracks"] = len(current_track_ids)
            metrics_dict["total_detections"] = total_detections
            metrics_dict["id_switches"] = id_switches
            metrics_dict["detections_by_class"] = dict(detection_by_class)
            metrics_dict["frame_count"] = frame_id

        if frame_callback is not None:
            frame_callback(frame)

        # Log stats every 100 frames
        if frame_id % 100 == 0:
            print(f"\n{'=' * 60}")
            print(f"Frame {frame_id} Statistics:")
            print(f"  FPS: {current_fps:.2f}")
            print(f"  Avg Inference: {avg_inference_ms:.2f}ms")
            print(f"  Avg Tracking: {avg_tracking_ms:.2f}ms")
            print(f"  Active Tracks: {len(current_track_ids)}")
            print(f"  Total Detections: {total_detections}")
            print(f"  ID Switches: {id_switches}")
            print(f"  Vehicle Counts: {dict(counts)}")
            print(f"{'=' * 60}\n")

        if display:
            cv2.imshow("Traffic Flow", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="test_video2.mp4")
    args = parser.parse_args()
    run_traffic_analysis(source=args.source)
