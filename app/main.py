"""
main.py
FastAPI tabanlı REST API — web paneline JSON çıktısı verir.

Endpoint'ler
------------
POST /analyze/image   — Tek görüntü analizi
POST /analyze/video   — Video dosyası analizi
GET  /stream/start    — Kamera/RTSP stream başlat
GET  /stream/stop     — Stream durdur
GET  /counts          — Anlık sayım sonuçları
GET  /health          — Servis sağlık kontrolü
"""

import asyncio
import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.tracker import ObjectTracker

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

tracker: Optional[ObjectTracker] = None
stream_active = False
latest_result = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tracker
    tracker = ObjectTracker(
        model_path="yolov8n.pt",
        conf_threshold=0.4,
        count_line_ratio=0.5,
        device="cpu",
    )
    print("✅ Model yüklendi, API hazır.")
    yield
    print("⛔ API kapatılıyor.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vision Counter API",
    description="YOLOv8 + ByteTrack tabanlı nesne takip ve sayım sistemi",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CountResponse(BaseModel):
    counts_per_class: dict
    total_count: int
    timestamp: float


class AnalysisResponse(BaseModel):
    frame_id: int
    timestamp: float
    detections: list
    counts_per_class: dict
    total_count: int
    fps: float
    annotated_image_b64: Optional[str] = None


class VideoSummary(BaseModel):
    total_frames_processed: int
    duration_seconds: float
    counts_per_class: dict
    total_count: int
    processing_fps: float


class StreamConfig(BaseModel):
    source: str = "0"                    # "0" = webcam, "rtsp://..." = IP kamera
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.4
    count_line_ratio: float = 0.5
    target_classes: Optional[list[str]] = None
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def decode_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Geçersiz görüntü dosyası.")
    return frame


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tracker is not None}


@app.get("/counts", response_model=CountResponse)
async def get_counts():
    """Anlık toplam sayım sonuçlarını döndür."""
    if tracker is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi.")
    return CountResponse(
        counts_per_class=dict(tracker.counts),
        total_count=len(tracker._counted_ids),
        timestamp=time.time(),
    )


@app.post("/counts/reset")
async def reset_counts():
    """Sayım geçmişini sıfırla."""
    if tracker is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi.")
    tracker.reset()
    return {"message": "Sayım sıfırlandı."}


@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    return_image: bool = Form(False),
):
    """
    Tek görüntü yükle, analiz et, JSON sonuç döndür.
    return_image=true ise annotated görüntü base64 olarak da döner.
    """
    if tracker is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi.")

    contents = await file.read()
    frame = decode_upload(contents)

    result = tracker.process_frame(frame)

    b64 = None
    if return_image:
        annotated = tracker.draw(frame.copy(), result)
        b64 = frame_to_b64(annotated)

    return AnalysisResponse(
        frame_id=result.frame_id,
        timestamp=result.timestamp,
        detections=result.detections,
        counts_per_class=result.counts_per_class,
        total_count=result.total_count,
        fps=result.fps,
        annotated_image_b64=b64,
    )


@app.post("/analyze/video", response_model=VideoSummary)
async def analyze_video(
    file: UploadFile = File(...),
    skip_frames: int = Form(2),
):
    """
    Video dosyası yükle, tüm kareleri işle, özet sayım döndür.
    skip_frames: Her N karede bir işle (hız için).
    """
    if tracker is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi.")

    tracker.reset()

    contents = await file.read()
    tmp_path = f"/tmp/upload_{int(time.time())}.mp4"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Video açılamadı.")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % (skip_frames + 1) == 0:
            tracker.process_frame(frame)
            processed += 1
        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    return VideoSummary(
        total_frames_processed=processed,
        duration_seconds=round(total_video_frames / fps_video, 2),
        counts_per_class=dict(tracker.counts),
        total_count=len(tracker._counted_ids),
        processing_fps=round(processed / max(elapsed, 0.001), 1),
    )


@app.post("/stream/configure")
async def configure_stream(config: StreamConfig):
    """Stream başlamadan önce tracker'ı yeniden yapılandır."""
    global tracker
    tracker = ObjectTracker(
        model_path=config.model_path,
        conf_threshold=config.conf_threshold,
        count_line_ratio=config.count_line_ratio,
        target_classes=config.target_classes,
        device=config.device,
    )
    return {"message": "Tracker yeniden yapılandırıldı.", "config": config.dict()}


@app.get("/stream/mjpeg")
async def mjpeg_stream(source: str = "0"):
    """
    MJPEG stream — tarayıcıda doğrudan <img src="/stream/mjpeg"> ile kullanılır.
    """
    global stream_active

    async def generate():
        global stream_active
        stream_active = True
        src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)

        try:
            while stream_active:
                ret, frame = cap.read()
                if not ret:
                    break

                result = tracker.process_frame(frame)
                annotated = tracker.draw(frame, result)

                _, buf = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                await asyncio.sleep(0.03)
        finally:
            cap.release()
            stream_active = False

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/stop")
async def stop_stream():
    global stream_active
    stream_active = False
    return {"message": "Stream durduruldu."}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
