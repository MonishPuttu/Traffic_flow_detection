from contextlib import asynccontextmanager
import threading
from fastapi import FastAPI
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
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Traffic Flow Viewer</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
        <style>
            :root {
                --bg: #f5f7fb;
                --surface: #ffffff;
                --surface-soft: #f9fbfd;
                --text-main: #17202f;
                --text-muted: #627087;
                --line: #dbe2ea;
                --line-strong: #c7d2de;
                --good: #2a7b61;
                --shadow: 0 12px 30px rgba(30, 50, 80, 0.08);
                --radius-lg: 20px;
                --radius-md: 14px;
                --radius-sm: 10px;
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                font-family: 'Manrope', sans-serif;
                background:
                    radial-gradient(circle at 15% -10%, #e9f0f2 0%, transparent 45%),
                    radial-gradient(circle at 100% 0%, #ebf1f8 0%, transparent 42%),
                    var(--bg);
                color: var(--text-main);
                padding: 18px 12px 18px;
            }

            .container {
                width: min(96vw, 1560px);
                margin: 0 auto;
            }

            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 18px;
                margin-bottom: 16px;
                flex-wrap: wrap;
            }

            .title-wrap h1 {
                margin: 0;
                font-size: clamp(1.8rem, 2.6vw, 2.75rem);
                letter-spacing: -0.03em;
                line-height: 1.15;
            }

            .title-wrap p {
                margin: 7px 0 0;
                color: var(--text-muted);
                font-size: 1.05rem;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                border: 1px solid var(--line);
                background: var(--surface);
                color: var(--text-main);
                border-radius: 999px;
                padding: 10px 14px;
                font-weight: 600;
                box-shadow: var(--shadow);
            }

            .status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: var(--good);
                box-shadow: 0 0 0 4px rgba(42, 123, 97, 0.15);
            }

            .layout {
                display: grid;
                gap: 20px;
                grid-template-columns: minmax(880px, 2.35fr) minmax(340px, 1fr);
                align-items: stretch;
            }

            .panel {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: var(--radius-lg);
                box-shadow: var(--shadow);
                overflow: hidden;
                height: 100%;
            }

            .video-container {
                padding: 16px;
            }

            .video-frame {
                position: relative;
                border: 1px solid var(--line-strong);
                background: #0f1624;
                border-radius: var(--radius-md);
                overflow: hidden;
            }

            img {
                width: 100%;
                display: block;
            }

            .video-meta {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 10px;
                margin-top: 14px;
                color: var(--text-muted);
                font-size: 0.95rem;
                flex-wrap: wrap;
            }

            .metrics-grid {
                display: grid;
                gap: 14px;
                grid-template-columns: 1fr 1fr;
                padding: 16px;
                grid-auto-rows: minmax(112px, 1fr);
                align-content: stretch;
            }

            .layout > aside.panel {
                display: flex;
                flex-direction: column;
            }

            .metric-card {
                background: var(--surface-soft);
                border: 1px solid var(--line);
                border-radius: var(--radius-sm);
                padding: 14px;
                min-height: 112px;
            }

            .metric-label {
                font-size: 0.78rem;
                color: var(--text-muted);
                letter-spacing: 0.03em;
                text-transform: uppercase;
                margin-bottom: 8px;
                font-weight: 700;
            }

            .metric-value {
                font-size: 1.7rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                color: var(--text-main);
                line-height: 1.15;
            }

            .metric-value.small {
                font-size: 1.08rem;
                font-weight: 600;
                line-height: 1.45;
                color: #30425f;
            }

            .metrics-header {
                border-bottom: 1px solid var(--line);
                padding: 14px 14px 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 10px;
            }

            .metrics-header h2 {
                margin: 0;
                font-size: 1.08rem;
                letter-spacing: -0.01em;
            }

            .refresh-info {
                margin: 0;
                color: var(--text-muted);
                font-size: 0.84rem;
                font-weight: 600;
                white-space: nowrap;
            }

            .metrics-empty {
                padding: 18px 14px 20px;
                color: var(--text-muted);
                font-size: 0.92rem;
            }

            @media (max-width: 1080px) {
                .layout {
                    grid-template-columns: 1fr;
                }

                .container {
                    width: min(97vw, 1160px);
                }
            }

            @media (max-width: 760px) {
                body {
                    padding-top: 18px;
                }

                .metrics-grid {
                    grid-template-columns: 1fr;
                }

                .video-container,
                .metrics-grid,
                .metrics-header {
                    padding-left: 12px;
                    padding-right: 12px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header class="header">
                <div class="title-wrap">
                    <h1>Traffic Flow Monitoring</h1>
                    <p>Live roadway analysis with low-latency performance telemetry</p>
                </div>
                <div class="status-pill">
                    <span class="status-dot"></span>
                    <span>Live Stream</span>
                </div>
            </header>

            <div class="layout">
                <section class="panel video-container">
                    <div class="video-frame">
                        <img src="/video_feed" alt="Traffic Analysis Feed" />
                    </div>
                    <div class="video-meta">
                        <span>Endpoint: /video_feed</span>
                        <span id="last-updated">Waiting for metrics...</span>
                    </div>
                </section>

                <aside class="panel">
                    <div class="metrics-header">
                        <h2>Performance Metrics</h2>
                        <p class="refresh-info">Auto-refresh: 2s</p>
                    </div>

                    <div class="metrics-grid" id="metrics">
                        <div class="metrics-empty">Loading metrics...</div>
                    </div>
                </aside>
            </div>
        </div>

        <script>
            function formatNumber(value, decimals = 2) {
                if (value === null || value === undefined || Number.isNaN(Number(value))) {
                    return '0';
                }

                const num = Number(value);
                return Number.isInteger(num)
                    ? String(num)
                    : num.toFixed(decimals);
            }

            function buildMetricCard(label, value, small = false) {
                return `
                    <div class="metric-card">
                        <div class="metric-label">${label}</div>
                        <div class="metric-value ${small ? 'small' : ''}">${value}</div>
                    </div>
                `;
            }

            async function updateMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();

                    const vehicleCounts = Object.entries(data.counts || {})
                        .map(([key, value]) => `${key}: ${value}`)
                        .join('<br>') || 'No vehicles counted';

                    const metricsHtml = [
                        buildMetricCard('FPS', formatNumber(data.fps, 1)),
                        buildMetricCard('Inference Time', `${formatNumber(data.inference_time_ms)} ms`),
                        buildMetricCard('Tracking Time', `${formatNumber(data.tracking_time_ms)} ms`),
                        buildMetricCard('Active Tracks', formatNumber(data.active_tracks, 0)),
                        buildMetricCard('Total Detections', formatNumber(data.total_detections, 0)),
                        buildMetricCard('ID Switches', formatNumber(data.id_switches, 0)),
                        buildMetricCard('Frames Processed', formatNumber(data.frame_count, 0)),
                        buildMetricCard('Vehicle Counts', vehicleCounts, true)
                    ].join('');

                    document.getElementById('metrics').innerHTML = metricsHtml;
                    document.getElementById('last-updated').textContent =
                        `Updated: ${new Date().toLocaleTimeString()}`;
                } catch (error) {
                    console.error('Error fetching metrics:', error);
                    document.getElementById('last-updated').textContent = 'Metrics unavailable';
                }
            }

            // Update metrics every 2 seconds
            updateMetrics();
            setInterval(updateMetrics, 2000);
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "fps": metrics.get("fps", 0.0),
        "active_tracks": metrics.get("active_tracks", 0)
    }