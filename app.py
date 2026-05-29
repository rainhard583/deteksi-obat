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

st.set_page_config(page_title="Deteksi Obat Realtime")

st.title("Deteksi Obat Realtime AI")


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_predictions = []
        self.last_request = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        now = time.time()

        # kirim frame ke Roboflow setiap 0.5 detik
        if now - self.last_request > 0.5:

            self.last_request = now

            small_img = cv2.resize(img, (320, 320))

            success, buffer = cv2.imencode(".jpg", small_img)

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
                        timeout=10
                    )

                    result = response.json()

                    self.last_predictions = result.get(
                        "predictions",
                        []
                    )

                except Exception:
                    pass

        obat_bagus = 0
        obat_rusak = 0

        for pred in self.last_predictions:

            confidence = pred.get("confidence", 0)

            if confidence < 0.70:
                continue

            kelas = pred["class"]

            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])

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
                0.5,
                warna,
                2
            )

        cv2.putText(
            img,
            f"Bagus: {obat_bagus}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Rusak: {obat_rusak}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
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