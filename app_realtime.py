
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import requests
import numpy as np
import time

API_KEY = "uc5sGP3TXol5G8dbrmo3"

ROBOFLOW_URL = (
    f"https://detect.roboflow.com/pills-deteks/3?api_key={API_KEY}"
)

st.set_page_config(page_title="Deteksi Obat Realtime")

st.title("REALTIME WEBSOCKET TEST")

counter_placeholder = st.empty()

class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_request = 0
        self.last_predictions = []
        self.obat_bagus = 0
        self.obat_rusak = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        now = time.time()

        # kirim ke Roboflow setiap 1 detik
        if now - self.last_request >= 1:

            self.last_request = now

            _, buffer = cv2.imencode(".jpg", img)

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
                    timeout=10
                )

                result = response.json()

                self.last_predictions = result.get(
                    "predictions",
                    []
                )

            except Exception:
                pass

        self.obat_bagus = 0
        self.obat_rusak = 0

        for pred in self.last_predictions:

            confidence = pred.get("confidence", 0)

            if confidence < 0.70:
                continue

            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])

            kelas = pred["class"]

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)

            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            if kelas == "obat_bagus":
                warna = (0, 255, 0)
                self.obat_bagus += 1
            else:
                warna = (0, 0, 255)
                self.obat_rusak += 1

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                warna,
                2
            )

            cv2.putText(
                img,
                kelas,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                warna,
                2
            )

        cv2.putText(
            img,
            f"Bagus: {self.obat_bagus}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Rusak: {self.obat_rusak}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        counter_placeholder.markdown(
            f"""
### Counter

🟢 Obat Bagus: **{self.obat_bagus}**

🔴 Obat Rusak: **{self.obat_rusak}**
"""
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

webrtc_streamer(
    key="deteksi-obat",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

