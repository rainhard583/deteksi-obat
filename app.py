import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import requests
import time

API_KEY = "sbCWG60zWcTHvD7PhnUY"

ROBOFLOW_URL = (
    f"https://detect.roboflow.com/pills-deteks-ye04t/2?api_key={API_KEY}"
)

st.set_page_config(page_title="Deteksi Obat Realtime")

st.title("Deteksi Obat Realtime AI")

# ==========================
# PILIH KAMERA
# ==========================

if "facing_mode" not in st.session_state:
    st.session_state.facing_mode = "environment"

col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Kamera Belakang", use_container_width=True):
        st.session_state.facing_mode = "environment"
with col2:
    if st.button("🤳 Kamera Depan", use_container_width=True):
        st.session_state.facing_mode = "user"

facing_mode = st.session_state.facing_mode
stream_key = f"deteksi-obat-{facing_mode}"

if facing_mode == "environment":
    st.info("📷 Menggunakan Kamera Belakang")
else:
    st.info("🤳 Menggunakan Kamera Depan")


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_predictions = []
        self.last_request = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        orig_h, orig_w = img.shape[:2]

        now = time.time()

        if now - self.last_request > 0.3:

            self.last_request = now

            small_img = cv2.resize(
                img,
                (224, 224)
            )

            success, buffer = cv2.imencode(
                ".jpg",
                small_img
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

                    result = response.json()
                    print("RESULT =", result)
                    self.last_predictions = result.get(
                        "predictions",
                        []
                    )

                except Exception:
                    pass

        obat_bagus = 0
        obat_rusak = 0

        scale_x = orig_w / 224
        scale_y = orig_h / 224

        for pred in self.last_predictions:

            confidence = pred.get("confidence", 0)

            if confidence < 0.85:
                continue

            kelas = pred["class"]

            x = int(pred["x"] * scale_x)
            y = int(pred["y"] * scale_y)

            w = int(pred["width"] * scale_x)
            h = int(pred["height"] * scale_y)

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)

            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            if kelas == "obat_bagus":

                warna = (0, 255, 0)
                obat_bagus += 1

            elif kelas == "obat_rusak":

                warna = (0, 0, 255)
                obat_rusak += 1

            else:

                warna = (255, 255, 0)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                warna,
                2
            )

            cv2.putText(
                img,
                f"{kelas} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                warna,
                2
            )

        cv2.putText(
            img,
            f"Bagus: {obat_bagus}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Rusak: {obat_rusak}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if obat_rusak > 0:

            cv2.rectangle(
                img,
                (0, 0),
                (orig_w, orig_h),
                (0, 0, 255),
                12
            )

            overlay = img.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (orig_w, orig_h),
                (0, 0, 255),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.15,
                img,
                0.85,
                0,
                img
            )

            cv2.putText(
                img,
                "OBAT RUSAK!",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (255, 255, 255),
                5
            )

            cv2.putText(
                img,
                "REJECT",
                (80, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                6
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key=stream_key,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": facing_mode}
        },
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)