# Vision Counter

**YOLOv8 + ByteTrack** tabanlı gerçek zamanlı nesne takip ve sayım sistemi.  
Web paneline bağlanmaya hazır **FastAPI** REST API çıktısı sunar.

---

## Özellikler

- 🎯 **YOLOv8 / YOLOv10** ile nesne tespiti
- 🔁 **ByteTrack** ile çok nesne takibi
- 📊 Çizgi geçişi ile doğru sayım
- 🌐 **FastAPI** REST API — web paneline JSON çıktısı
- 📷 Webcam, video dosyası ve RTSP kamera desteği
- 🔧 Sınıf filtresi (sadece insan, araç, vb. sayılabilir)

---

## Kurulum

### Gereksinimler
- Python 3.10+
- pip

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI_ADIN/vision-counter.git
cd vision-counter

# 2. Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

> Model dosyası (`yolov8n.pt`) ilk çalıştırmada otomatik indirilir.

---

## Kullanım

### 1. Demo — Webcam veya Video

```bash
# Webcam
python run_demo.py

# Video dosyası
python run_demo.py --source video.mp4

# RTSP kamera
python run_demo.py --source rtsp://192.168.1.100:554/stream

# Sadece insan ve araç say
python run_demo.py --classes person car

# GPU kullan
python run_demo.py --device cuda
```

| Tuş | İşlev |
|-----|-------|
| `Q` | Çıkış |
| `R` | Sayımı sıfırla |

---

### 2. API Sunucusu

```bash
python -m app.main
# veya
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Tarayıcıda aç: http://localhost:8000/docs

---

## API Endpoint'leri

### `GET /health`
Servis sağlık kontrolü.
```json
{"status": "ok", "model_loaded": true}
```

---

### `POST /analyze/image`
Tek görüntü analizi.

**Form-data:**
| Alan | Tip | Açıklama |
|------|-----|----------|
| `file` | dosya | JPG / PNG görüntü |
| `return_image` | bool | true ise annotated görüntü base64 döner |

**Yanıt:**
```json
{
  "frame_id": 1,
  "timestamp": 1720000000.0,
  "detections": [
    {
      "track_id": 3,
      "class_name": "person",
      "confidence": 0.87,
      "bbox": [120, 80, 300, 450],
      "center": [210, 265],
      "crossed_line": true
    }
  ],
  "counts_per_class": {"person": 2, "car": 1},
  "total_count": 3,
  "fps": 24.5,
  "annotated_image_b64": "..."
}
```

---

### `POST /analyze/video`
Video dosyası analizi — özet sayım döndürür.

**Form-data:** `file` (mp4/avi), `skip_frames` (int, varsayılan: 2)

**Yanıt:**
```json
{
  "total_frames_processed": 420,
  "duration_seconds": 28.4,
  "counts_per_class": {"person": 15, "car": 7},
  "total_count": 22,
  "processing_fps": 18.3
}
```

---

### `GET /counts`
Anlık toplam sayım.
```json
{
  "counts_per_class": {"person": 5},
  "total_count": 5,
  "timestamp": 1720000000.0
}
```

---

### `POST /counts/reset`
Sayımı sıfırla.

---

### `GET /stream/mjpeg?source=0`
MJPEG canlı stream — tarayıcıda doğrudan görüntülenebilir:
```html
<img src="http://localhost:8000/stream/mjpeg?source=0" />
```

---

### `POST /stream/configure`
Tracker parametrelerini güncelle:
```json
{
  "source": "rtsp://...",
  "model_path": "yolov8s.pt",
  "conf_threshold": 0.5,
  "count_line_ratio": 0.6,
  "target_classes": ["person", "car"],
  "device": "cpu"
}
```

---

### `GET /stream/stop`
Aktif stream'i durdur.

---

## Proje Yapısı

```
vision-counter/
├── app/
│   ├── main.py          # FastAPI uygulaması
│   └── tracker.py       # YOLOv8 + ByteTrack sayım motoru
├── run_demo.py          # CLI demo scripti
├── requirements.txt
└── README.md
```

---

## Teknik Detaylar

| Bileşen | Teknoloji |
|---------|-----------|
| Nesne tespiti | YOLOv8 / YOLOv10 (Ultralytics) |
| Nesne takibi | ByteTrack (YOLO'ya entegre) |
| Sayım yöntemi | Sanal çizgi geçiş tespiti |
| API | FastAPI + Uvicorn |
| Görüntü işleme | OpenCV |

---

## Lisans

MIT
