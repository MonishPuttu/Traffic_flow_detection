from contextlib import asynccontextmanager
import threading
from io import BytesIO
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from traffic_analysis import run_traffic_analysis
from prometheus_client import Gauge, Counter, Histogram, generate_latest, REGISTRY
import cv2
import time

# Enhanced Prometheus metrics
vehicle_count = Gauge("vehicle_count_total", "Total vehicle count", ["type"])
fps_metric = Gauge("fps", "Current frames per second")
inference_time = Gauge("inference_time_ms", "Average inference time in milliseconds")
tracking_time = Gauge("tracking_time_ms", "Average tracking time in milliseconds")
active_tracks = Gauge("active_tracks", "Number of currently active vehicle tracks")
total_detections = Counter("total_detections", "Total number of vehicle detections")
id_switches = Counter("id_switches_total", "Total number of track ID switches")
frame_count = Counter("frame_count_total", "Total frames processed")
detections_by_class = Gauge("detections_by_class", "Total detections per vehicle class", ["class"])

# Processing time histogram for detailed analysis
processing_histogram = Histogram(
    "frame_processing_seconds",
    "Frame processing time distribution",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

# shared metrics dict between threads
metrics = {
    "counts": {}, 
    "fps": 0.0,
    "inference_time_ms": 0.0,
    "tracking_time_ms": 0.0,
    "active_tracks": 0,
    "total_detections": 0,
    "id_switches": 0,
    "detections_by_class": {},
    "frame_count": 0
}

# Global variable to store latest processed frame
latest_frame = None
frame_lock = threading.Lock()

def analysis_worker():
    """Run YOLO + DeepSORT in a background thread."""
    global latest_frame
    run_traffic_analysis(
        source="test_video2.mp4",
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
    
    # Shutdown: Cleanup
    print("🔴 Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Traffic Flow Analysis API", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Traffic Flow Analysis API is running"}

@app.get("/metrics")
def get_metrics():
    """Return latest metrics (counts, fps, performance stats)."""
    return metrics

@app.get("/prometheus")
def prometheus_metrics():
    """Expose metrics in Prometheus format."""
    # Update all Prometheus metrics
    for cls, val in metrics.get("counts", {}).items():
        vehicle_count.labels(type=cls).set(val)
    
    fps_metric.set(metrics.get("fps", 0.0))
    inference_time.set(metrics.get("inference_time_ms", 0.0))
    tracking_time.set(metrics.get("tracking_time_ms", 0.0))
    active_tracks.set(metrics.get("active_tracks", 0))
    
    # Update counters (only if they've increased)
    current_detections = metrics.get("total_detections", 0)
    current_id_switches = metrics.get("id_switches", 0)
    current_frame_count = metrics.get("frame_count", 0)
    
    # Set counter values directly (Prometheus will track increases)
    total_detections._value._value = current_detections
    id_switches._value._value = current_id_switches
    frame_count._value._value = current_frame_count
    
    # Update class-specific detections - FIXED: using 'class' instead of 'class_name'
    for cls, val in metrics.get("detections_by_class", {}).items():
        detections_by_class.labels(**{"class": cls}).set(val)
    
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
                    time.sleep(0.1)
            time.sleep(0.033)  # ~30 fps
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    """HTML page to view the video stream with metrics."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚗 Real-Time Traffic Flow Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #1a1a1a;
                color: #00ff00;
                padding: 20px;
                margin: 0;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #00ff00;
            }
            .video-container {
                text-align: center;
                margin: 20px 0;
                border: 3px solid #00ff00;
                border-radius: 10px;
                overflow: hidden;
            }
            img {
                width: 100%;
                max-width: 1200px;
                display: block;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .metric-card {
                background: #2a2a2a;
                border: 2px solid #00ff00;
                border-radius: 8px;
                padding: 15px;
            }
            .metric-label {
                font-size: 14px;
                color: #888;
                margin-bottom: 5px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #00ff00;
            }
            .refresh-info {
                text-align: center;
                margin-top: 10px;
                color: #666;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Real-Time Traffic Flow Analysis</h1>
            
            <div class="video-container">
                <img src="/video_feed" alt="Traffic Analysis Feed">
            </div>
            
            <div class="metrics-grid" id="metrics">
                <!-- Metrics will be populated by JavaScript -->
            </div>
            
            <div class="refresh-info">
                Metrics auto-refresh every 2 seconds
            </div>
        </div>
        
        <script>
            async function updateMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    
                    const metricsHtml = `
                        <div class="metric-card">
                            <div class="metric-label">FPS</div>
                            <div class="metric-value">${data.fps || 0}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Inference Time</div>
                            <div class="metric-value">${data.inference_time_ms || 0} ms</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Tracking Time</div>
                            <div class="metric-value">${data.tracking_time_ms || 0} ms</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Active Tracks</div>
                            <div class="metric-value">${data.active_tracks || 0}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Total Detections</div>
                            <div class="metric-value">${data.total_detections || 0}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">ID Switches</div>
                            <div class="metric-value">${data.id_switches || 0}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Frames Processed</div>
                            <div class="metric-value">${data.frame_count || 0}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Vehicle Counts</div>
                            <div class="metric-value" style="font-size: 16px;">
                                ${Object.entries(data.counts || {})
                                    .map(([k, v]) => `${k}: ${v}`)
                                    .join('<br>')}
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('metrics').innerHTML = metricsHtml;
                } catch (error) {
                    console.error('Error fetching metrics:', error);
                }
            }
            
            // Update metrics every 2 seconds
            updateMetrics();
            setInterval(updateMetrics, 2000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "fps": metrics.get("fps", 0.0),
        "active_tracks": metrics.get("active_tracks", 0)
    }
