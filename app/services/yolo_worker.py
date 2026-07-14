import cv2
import sys
import torch
from ultralytics import YOLO

def test_yolo(stream_url):
    print("🔥 Memanaskan mesin YOLOv8...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n.pt")
    model.to(device)

    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ Gagal buka stream CCTV!")
        return

    print("✅ Berhasil buka CCTV! Mulai deteksi pakai Frame Skipping...")

    frame_count = 0
    skip_rate = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Stream terputus!")
            break

        frame_count += 1

        if frame_count % skip_rate == 0:
            print(f"📸 Menganalisis Frame ke-{frame_count}...")

            results = model(frame, classes=[2, 3, 5, 7], verbose=False, device=device)

            boxes = results[0].boxes
            print(f"🎯 KETEMU {len(boxes)} OBJEK DI FRAME {frame_count}!")

            if frame_count >= 20:
                break

    cap.release()
    print("🏁 Tes Selesai!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    else:
        target_url = "https://livepantau.semarangkota.go.id/a875df34-d235-4760-8c7f-2705fb155807/index.m3u8"
        print(f"⚠️ URL tidak diberikan via parameter, menggunakan default: {target_url}")

    test_yolo(target_url)

