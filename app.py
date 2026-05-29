import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import requests
import time

API_KEY = "uc5sGP3TXol5G8dbrmo3"

ROBOFLOW_URL = (
    f"https://detect.roboflow.com/pills-deteks/3?api_key={API_KEY}"
)

# =============================================
# CONFIG
# =============================================
RESIZE_DIM = 640        # Naik dari 224 → 640 (lebih akurat)
CONFIDENCE_MIN = 0.40   # Turun dari 0.70 → 0.40 (lebih sensitif)
REQUEST_INTERVAL = 0.8  # Interval request (detik)

st.set_page_config(page_title="Deteksi Obat Realtime", page_icon="💊")

st.title("💊 Deteksi Obat Realtime AI")
st.caption(f"Model: pills-deteks/3 | Min. confidence: {int(CONFIDENCE_MIN*100)}% | Resolusi deteksi: {RESIZE_DIM}px")


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_predictions = []
        self.last_request = 0
        self.last_error = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = img.shape[:2]

        now = time.time()

        if now - self.last_request > REQUEST_INTERVAL:

            self.last_request = now

            # FIX 1: Naikkan resolusi dari 224 → 640
            small_img = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))

            success, buffer = cv2.imencode(
                ".jpg",
                small_img,
                [cv2.IMWRITE_JPEG_QUALITY, 90]  # FIX 2: Kualitas JPEG lebih tinggi
            )

            if success:
                try:
                    response = requests.post(
                        ROBOFLOW_URL,
                        files={
                            "file": (
                                "frame.jpg",
                                buffer.tobytes(),
                                "image/jpeg"
                            )
                        },
                        timeout=5
                    )

                    if response.status_code == 200:
                        result = response.json()
                        self.last_predictions = result.get("predictions", [])
                        self.last_error = None
                        print(f"[OK] {len(self.last_predictions)} prediksi diterima")
                    else:
                        self.last_error = f"HTTP {response.status_code}"
                        print(f"[ERROR] Roboflow HTTP {response.status_code}: {response.text}")

                except requests.exceptions.Timeout:
                    self.last_error = "Timeout"
                    print("[ERROR] Request timeout")
                except Exception as e:
                    self.last_error = str(e)
                    print(f"[ERROR] Exception: {e}")

        obat_bagus = 0
        obat_rusak = 0

        # FIX 3: Skala koordinat disesuaikan dengan RESIZE_DIM baru
        scale_x = orig_w / RESIZE_DIM
        scale_y = orig_h / RESIZE_DIM

        for pred in self.last_predictions:

            confidence = pred.get("confidence", 0)

            # FIX 4: Threshold lebih rendah → lebih banyak deteksi tertangkap
            if confidence < CONFIDENCE_MIN:
                continue

            kelas = pred["class"]

            x = pred["x"] * scale_x
            y = pred["y"] * scale_y
            w = pred["width"] * scale_x
            h = pred["height"] * scale_y

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            if kelas == "obat_bagus":
                warna = (0, 220, 50)
                obat_bagus += 1
            elif kelas == "obat_rusak":
                warna = (0, 0, 220)
                obat_rusak += 1
            else:
                warna = (255, 200, 0)

            # Gambar bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), warna, 2)

            # Label dengan background biar lebih terbaca
            label = f"{kelas} {confidence:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw + 4, y1), warna, -1)
            cv2.putText(
                img, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2
            )

        # Counter di pojok kiri atas
        cv2.rectangle(img, (0, 0), (200, 90), (20, 20, 20), -1)
        cv2.putText(img, f"Bagus: {obat_bagus}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 50), 2)
        cv2.putText(img, f"Rusak: {obat_rusak}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 220), 2)

        # Tampilkan error di frame kalau ada
        if self.last_error:
            err_text = f"ERR: {self.last_error}"
            cv2.putText(img, err_text, (10, orig_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        # ==============================
        # ALARM VISUAL — OBAT RUSAK
        # ==============================
        if obat_rusak > 0:

            # Border merah tebal
            cv2.rectangle(img, (0, 0), (orig_w, orig_h), (0, 0, 220), 14)

            # Overlay merah transparan
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (orig_w, orig_h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

            # Teks peringatan
            cv2.putText(img, "OBAT RUSAK!", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
            cv2.putText(img, "REJECT", (80, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 6)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =============================================
# WEBRTC STREAMER
# =============================================
webrtc_streamer(
    key="deteksi-obat",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": "environment"},
            "width": {"ideal": 1280},   # FIX 5: Minta resolusi kamera lebih tinggi
            "height": {"ideal": 720}
        },
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}  # FIX 6: Tambah STUN backup
        ]
    }
)