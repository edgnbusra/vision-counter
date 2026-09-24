"""
run_demo.py
Komut satırından webcam, video dosyası veya RTSP stream üzerinde
nesne takibi ve sayımı yapar.

Örnekler
--------
  python run_demo.py                          # Webcam (kamera 0)
  python run_demo.py --source video.mp4       # Video dosyası
  python run_demo.py --source rtsp://...      # IP kamera
  python run_demo.py --classes person car     # Sadece insan ve araç say
  python run_demo.py --model yolov8s.pt       # Farklı model
"""

import argparse
import time

import cv2

from app.tracker import ObjectTracker


def parse_args():
    p = argparse.ArgumentParser(description="Vision Counter Demo")
    p.add_argument("--source", default="0", help="Video kaynağı: 0=webcam, dosya yolu, RTSP URL")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model dosyası")
    p.add_argument("--conf", type=float, default=0.4, help="Güven eşiği")
    p.add_argument("--line", type=float, default=0.5, help="Sayım çizgisi konumu (0-1)")
    p.add_argument("--classes", nargs="*", default=None, help="Takip edilecek sınıflar")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    p.add_argument("--no-display", action="store_true", help="Pencere gösterme (sunucu modu)")
    return p.parse_args()


def main():
    args = parse_args()

    tracker = ObjectTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        count_line_ratio=args.line,
        target_classes=args.classes,
        device=args.device,
    )

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Kaynak açılamadı: {args.source}")
        return

    print(f"✅ Kaynak açıldı: {args.source}")
    print("   [Q] Çıkış | [R] Sayımı Sıfırla")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame okunamadı, duruluyor.")
            break

        result = tracker.process_frame(frame)
        annotated = tracker.draw(frame, result)

        # Terminal çıktısı
        if result.frame_id % 30 == 0:
            print(
                f"Frame {result.frame_id:5d} | "
                f"FPS: {result.fps:5.1f} | "
                f"Toplam: {result.total_count} | "
                f"{result.counts_per_class}"
            )

        if not args.no_display:
            cv2.imshow("Vision Counter", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                tracker.reset()
                print("🔄 Sayım sıfırlandı.")

    cap.release()
    cv2.destroyAllWindows()

    print("\n📊 SONUÇLAR")
    print("-" * 30)
    print(f"Toplam sayılan: {result.total_count}")
    for cls, cnt in result.counts_per_class.items():
        print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    main()
