"""
tracker.py
YOLOv8 + ByteTrack tabanlı nesne takip ve sayım modülü.
Kamera stream, video dosyası veya tek kare görüntü destekler.
"""

import cv2
import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ultralytics import YOLO


@dataclass
class TrackRecord:
    track_id: int
    class_id: int
    class_name: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    counted: bool = False


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: list[dict]
    counts_per_class: dict[str, int]
    total_count: int
    fps: float


class ObjectTracker:
    """
    YOLOv8 inference + ByteTrack takip + çizgi geçişi ile nesne sayımı.

    Parametreler
    ----------
    model_path : str
        YOLOv8 / YOLOv10 ağırlık dosyası (.pt)
    conf_threshold : float
        Güven eşiği (0-1)
    iou_threshold : float
        NMS IoU eşiği
    count_line_ratio : float
        Sayım çizgisinin frame yüksekliğindeki konumu (0-1)
    target_classes : list[str] | None
        Takip edilecek sınıflar; None ise tüm sınıflar
    device : str
        'cpu', 'cuda', 'mps'
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        count_line_ratio: float = 0.5,
        target_classes: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.line_ratio = count_line_ratio
        self.target_classes = target_classes
        self.device = device

        # Takip geçmişi  {track_id: TrackRecord}
        self._records: dict[int, TrackRecord] = {}
        # Sayılan id'ler
        self._counted_ids: set[int] = set()
        # Sınıf bazlı toplam sayım
        self.counts: dict[str, int] = defaultdict(int)

        self._frame_id = 0
        self._prev_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Tek bir frame'i işle, takip et, say ve sonuçları döndür.
        """
        self._frame_id += 1
        h, w = frame.shape[:2]
        line_y = int(h * self.line_ratio)

        # YOLOv8 tracker inference (ByteTrack)
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )

        detections = []
        now = time.time()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                class_id = int(box.cls.item())
                class_name = self.model.names[class_id]

                # Sınıf filtresi
                if self.target_classes and class_name not in self.target_classes:
                    continue

                conf_score = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Kayıt güncelle
                if track_id not in self._records:
                    self._records[track_id] = TrackRecord(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        first_seen=now,
                    )
                self._records[track_id].last_seen = now

                # Çizgi geçişi kontrolü
                crossed = self._check_crossing(track_id, class_name, cy, line_y)

                detections.append(
                    {
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(conf_score, 3),
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                        "crossed_line": crossed,
                    }
                )

        fps = self._calc_fps()

        return FrameResult(
            frame_id=self._frame_id,
            timestamp=now,
            detections=detections,
            counts_per_class=dict(self.counts),
            total_count=len(self._counted_ids),
            fps=round(fps, 1),
        )

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """
        Bounding box, ID, sayım çizgisi ve sayaç bilgisini frame üzerine çizer.
        """
        h, w = frame.shape[:2]
        line_y = int(h * self.line_ratio)

        # Sayım çizgisi
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(
            frame, "SAYIM ÇIZGISI", (10, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        # Tespitler
        for det in result.detections:
            x1, y1, x2, y2 = det["bbox"]
            tid = det["track_id"]
            label = f"{det['class_name']} #{tid} ({det['confidence']:.2f})"
            color = (0, 200, 0) if det["crossed_line"] else (255, 100, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        # HUD — sağ üst köşe
        hud_lines = [
            f"FPS: {result.fps}",
            f"Toplam: {result.total_count}",
        ] + [f"{k}: {v}" for k, v in result.counts_per_class.items()]

        for i, text in enumerate(hud_lines):
            cv2.putText(
                frame, text, (w - 200, 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        return frame

    def reset(self):
        """Sayım ve takip geçmişini sıfırla."""
        self._records.clear()
        self._counted_ids.clear()
        self.counts = defaultdict(int)
        self._frame_id = 0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_crossing(
        self, track_id: int, class_name: str, cy: int, line_y: int
    ) -> bool:
        """
        Nesnenin sayım çizgisini geçip geçmediğini kontrol et.
        Merkez y koordinatı çizgiye ±20 px içindeyse say.
        """
        if track_id in self._counted_ids:
            return True

        if abs(cy - line_y) < 20:
            self._counted_ids.add(track_id)
            self.counts[class_name] += 1
            return True

        return False

    def _calc_fps(self) -> float:
        now = time.time()
        fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now
        return fps
