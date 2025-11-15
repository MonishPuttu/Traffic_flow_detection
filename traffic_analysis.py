import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import time


# CONFIG
CONF_THRESH = 0.35
COUNT_LINE = [(200, 400), (800, 400)]   # adjust to your video
CLASSES_TO_COUNT = ["car", "truck", "bus", "motorcycle"]
MODEL_WEIGHTS = "yolov8n.pt"  # tiny, fast model


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
def run_traffic_analysis(source="test_video.mp4", display=True, metrics_dict=None, frame_callback=None):
    model = YOLO(MODEL_WEIGHTS)
    tracker = DeepSort(max_age=30)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    track_memory = {}        # track_id -> last centroid
    counts = defaultdict(int)
    fps_time = time.time()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # Inference
        results = model.predict(frame, imgsz=640, conf=CONF_THRESH, verbose=False)
        dets = []
        for res in results:
            boxes = res.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if cls_name in CLASSES_TO_COUNT and conf >= CONF_THRESH:
                    dets.append(([x1, y1, x2, y2], conf, cls_name))

        # DeepSORT update
        ds_input = [ (b, conf, cls) for (b, conf, cls) in dets ]
        tracks = tracker.update_tracks(ds_input, frame=frame)

        # Process tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            c = centroid((x1, y1, x2, y2))
            cls_name = track.det_class if hasattr(track, "det_class") else "vehicle"

            if tid in track_memory:
                prev_c = track_memory[tid]
                if segments_intersect(prev_c, c, COUNT_LINE[0], COUNT_LINE[1]):
                    counts[cls_name] += 1

            track_memory[tid] = c
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name}-{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, c, 3, (255, 0, 0), -1)

        # Draw counting line and metrics
        cv2.line(frame, COUNT_LINE[0], COUNT_LINE[1], (0, 0, 255), 2)
        cv2.putText(frame, "Counts: " + str(dict(counts)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # update shared metrics for API
        if metrics_dict is not None:
            metrics_dict["counts"] = dict(counts)
            metrics_dict["fps"] = round(fps, 2)

        if frame_callback is not None:
            frame_callback(frame)

        if display:
            cv2.imshow("Traffic Flow", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="test_video.mp4")
    args = parser.parse_args()
    run_traffic_analysis(source=args.source)
