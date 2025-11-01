import threading
import time
import base64
import cv2
from io import BytesIO
import asyncio
import json
from fastapi import FastAPI, WebSocket
from traffic_analysis import run_traffic_analysis

app = FastAPI(title="Traffic Flow Analysis API")

# shared metrics dict between threads
metrics = {"counts": {}, "fps": 0.0}

# optional: store latest frame
latest_frame = None

def analysis_worker():
    """Run YOLO + DeepSORT in a background thread."""
    global latest_frame
    run_traffic_analysis(
        source="test_video.mp4",
        display=False,
        metrics_dict=metrics
    )

@app.get("/")
def root():
    return {"message": "Traffic Flow Analysis API is running"}

@app.get("/metrics")
def get_metrics():
    """Return latest metrics (counts, fps)."""
    return metrics

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Push metrics to connected clients in real time."""
    await ws.accept()
    while True:
        await ws.send_text(json.dumps(metrics))
        await asyncio.sleep(1)

if __name__ == "__main__":
    t = threading.Thread(target=analysis_worker, daemon=True)
    t.start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
