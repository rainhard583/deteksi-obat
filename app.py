import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import requests
import time
import numpy as np

API_KEY = "uc5sGP3TXol5G8dbrmo3"
ROBOFLOW_URL = f"https://detect.roboflow.com/pills-deteks/3?api_key={API_KEY}"

st.set_page_config(
    page_title="Deteksi Obat AI",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0a0c10;
    color: white;
}
.metric-box{
    padding:15px;
    border-radius:10px;
    background:#151922;
    text-align:center;
}
.metric-title{
    color:#888;
    font-size:12px;
}
.metric-value{
    font-size:36px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.title("💊 Deteksi Obat AI Realtime")

cam_col, side_col = st.columns([3,1])

good_placeholder = None
bad_placeholder = None

with side_col:

    st.subheader("Dashboard")

    good_placeholder = st.empty()
    bad_placeholder = st.empty()

    st.info("Model: pills-deteks/v3")
    st.info("Confidence: 0.70")
    st.info("Realtime Roboflow API")


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_predictions = []
        self.last_request = 0

        self.good_count = 0
        self.bad_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        orig_h, orig_w = img.shape[:2]

        now = time.time()

        if now - self.last_request > 0.3:

            self.last_request = now

            small_img = cv2.resize(
                img,
                (224,224)
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

                    self.last_predictions = result.get(
                        "predictions",
                        []
                    )

                except Exception:
                    pass

        scale_x = orig_w / 224
        scale_y = orig_h / 224

        grid = np.zeros_like(img)

        grid[::32, :] = (40,42,60)
        grid[:, ::32] = (40,42,60)

        cv2.addWeighted(
            grid,
            0.12,
            img,
            0.88,
            0,
            img
        )

        good_count = 0
        bad_count = 0

        for pred in self.last_predictions:

            confidence = pred.get(
                "confidence",
                0
            )

            if confidence < 0.70:
                continue

            kelas = pred["class"]

            x = int(pred["x"] * scale_x)
            y = int(pred["y"] * scale_y)

            w = int(pred["width"] * scale_x)
            h = int(pred["height"] * scale_y)

            x1 = int(x - w/2)
            y1 = int(y - h/2)

            x2 = int(x + w/2)
            y2 = int(y + h/2)

            if kelas == "obat_bagus":

                warna = (0,255,0)
                good_count += 1

            elif kelas == "obat_rusak":

                warna = (0,0,255)
                bad_count += 1

            else:

                warna = (255,255,0)

            cv2.rectangle(
                img,
                (x1,y1),
                (x2,y2),
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

        self.good_count = good_count
        self.bad_count = bad_count

        cv2.putText(
            img,
            f"Bagus: {good_count}",
            (10,35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            img,
            f"Rusak: {bad_count}",
            (10,75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        if bad_count > 0:

            overlay = img.copy()

            cv2.rectangle(
                overlay,
                (0,0),
                (orig_w,orig_h),
                (0,0,255),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.10,
                img,
                0.90,
                0,
                img
            )

            cv2.rectangle(
                img,
                (0,0),
                (orig_w,orig_h),
                (0,0,255),
                5
            )

            cv2.putText(
                img,
                "REJECT",
                (50,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255,255,255),
                4
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


with cam_col:

    ctx = webrtc_streamer(
        key="deteksi-obat",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": {
                    "ideal": "environment"
                }
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

if ctx and hasattr(ctx, "video_processor") and ctx.video_processor:

    vp = ctx.video_processor

    good_placeholder.markdown(
        f"""
        <div class='metric-box'>
            <div class='metric-title'>OBAT BAGUS</div>
            <div class='metric-value' style='color:#00ff88'>
                {vp.good_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    bad_placeholder.markdown(
        f"""
        <div class='metric-box'>
            <div class='metric-title'>OBAT RUSAK</div>
            <div class='metric-value' style='color:#ff4444'>
                {vp.bad_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )