from contextlib import asynccontextmanager
import threading
from io import BytesIO
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from traffic_analysis import run_traffic_analysis
from prometheus_client import Gauge, generate_latest, REGISTRY
import cv2
import time

# Prometheus metrics
vehicle_count = Gauge("vehicle_count_total", "Total vehicle count", ["type"])
fps_metric = Gauge("fps", "Frames per second")

# shared metrics dict between threads
metrics = {"counts": {}, "fps": 0.0}

# Global variable to store latest processed frame
latest_frame = None
frame_lock = threading.Lock()

def analysis_worker():
    """Run YOLO + DeepSORT in a background thread."""
    global latest_frame
    run_traffic_analysis(
        source="test_video.mp4",
        display=False,
        metrics_dict=metrics,
        frame_callback=update_frame
    )

def update_frame(frame):
    """Callback to update the latest frame from analysis."""
    global latest_frame
    with frame_lock:
        latest_frame = frame.copy()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the video analysis thread
    thread = threading.Thread(target=analysis_worker, daemon=True)
    thread.start()
    print("✅ Analysis worker thread started")
    
    yield
    
    # Shutdown: Cleanup (optional)
    print("🔴 Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Traffic Flow Analysis API", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Traffic Flow Analysis API is running"}


@app.get("/metrics")
def get_metrics():
    """Return latest metrics (counts, fps)."""
    return metrics


@app.get("/prometheus")
def prometheus_metrics():
    # Update metrics before exposing them
    for cls, val in metrics.get("counts", {}).items():
        vehicle_count.labels(type=cls).set(val)
    fps_metric.set(metrics.get("fps", 0.0))

    # Convert to Prometheus format
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type="text/plain")


@app.get("/video_feed")
async def video_feed():
    """Stream processed video frames as MJPEG."""
    def generate():
      while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame, 
                                          [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           frame_bytes + b'\r\n')
            else:
                # Send a blank frame or wait if no frame available yet
                time.sleep(0.1)
                time.sleep(0.033)  # ~30 fps
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    """HTML page to view the video stream."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Flow Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #1a1a1a;
                color: #ffffff;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            h1 {
                color: #00ff88;
                margin-bottom: 20px;
            }
            .video-container {
                border: 3px solid #00ff88;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
                max-width: 1280px;
                width: 100%;
            }
            img {
                width: 100%;
                height: auto;
                display: block;
            }
            .metrics {
                margin-top: 20px;
                padding: 20px;
                background-color: #2a2a2a;
                border-radius: 10px;
                width: 100%;
                max-width: 1280px;
            }
            .metric-item {
                margin: 10px 0;
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <h1>🚗 Real-Time Traffic Flow Analysis</h1>
        <div class="video-container">
            <img src="/video_feed" alt="Traffic Analysis Stream">
        </div>
        <div class="metrics" id="metrics">
            <h2>Live Metrics</h2>
            <div id="metrics-data">Loading...</div>
        </div>
        
        <script>
            // Update metrics every second
            setInterval(async () => {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    const metricsDiv = document.getElementById('metrics-data');
                    
                    let html = '<div class="metric-item">FPS: ' + data.fps + '</div>';
                    html += '<h3>Vehicle Counts:</h3>';
                    
                    for (const [type, count] of Object.entries(data.counts || {})) {
                        html += '<div class="metric-item">' + 
                                type.charAt(0).toUpperCase() + type.slice(1) + 
                                ': ' + count + '</div>';
                    }
                    
                    metricsDiv.innerHTML = html;
                } catch (error) {
                    console.error('Error fetching metrics:', error);
                }
            }, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
