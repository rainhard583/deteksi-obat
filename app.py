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
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: #0a0c10; color: #e8eaf0; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1200px; }

.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; background: #111318;
    border: 1px solid #1f2230; border-radius: 12px; margin-bottom: 20px;
}
.app-title { font-size: 18px; font-weight: 600; color: #e8eaf0; letter-spacing: -0.3px; }
.app-subtitle { font-size: 12px; color: #4b5563; font-family: 'DM Mono', monospace; margin-top: 2px; }
.live-badge {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; font-family: 'DM Mono', monospace; color: #10d98a;
    background: rgba(16,217,138,0.08); border: 1px solid rgba(16,217,138,0.2);
    border-radius: 20px; padding: 5px 12px;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #10d98a;
    animation: pulse 1.4s ease-in-out infinite; display: inline-block;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }

.stat-card { background: #111318; border: 1px solid #1f2230; border-radius: 12px; padding: 16px 20px; text-align: center; }
.stat-label { font-size: 11px; font-family: 'DM Mono', monospace; letter-spacing: 1.5px; text-transform: uppercase; color: #4b5563; margin-bottom: 8px; }
.stat-value { font-family: 'DM Mono', monospace; font-size: 42px; font-weight: 500; line-height: 1; }
.stat-good { color: #10d98a; }
.stat-bad  { color: #ef4444; }

.alert-reject {
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.35);
    border-radius: 10px; padding: 14px 18px; display: flex; align-items: center;
    gap: 10px; font-family: 'DM Mono', monospace; font-size: 13px;
    letter-spacing: 1px; color: #ef4444; margin-top: 12px;
}
.alert-dot { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; animation: pulse 0.8s ease-in-out infinite; flex-shrink: 0; }
.alert-ok { background: rgba(16,217,138,0.06); border: 1px solid rgba(16,217,138,0.2); border-radius: 10px; padding: 14px 18px; font-family: 'DM Mono', monospace; font-size: 13px; color: #10d98a; margin-top: 12px; }

.progress-wrap { background: #1a1d28; border-radius: 4px; height: 5px; overflow: hidden; margin-top: 12px; }
.progress-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }

.info-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.info-table td { padding: 6px 0; border-bottom: 1px solid #1a1d28; }
.info-table td:first-child { color: #4b5563; font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }
.info-table td:last-child { color: #9ca3af; text-align: right; font-family: 'DM Mono', monospace; }
.info-table span.accent { color: #6366f1; }

.section-header { font-size: 10px; font-family: 'DM Mono', monospace; letter-spacing: 1.5px; text-transform: uppercase; color: #4b5563; margin-bottom: 12px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div>
    <div class="app-title">💊 Deteksi Obat AI</div>
    <div class="app-subtitle">pills-deteks/v3 · Roboflow · confidence ≥ 0.70</div>
  </div>
  <div class="live-badge"><span class="live-dot"></span>REALTIME</div>
</div>
""", unsafe_allow_html=True)

cam_col, side_col = st.columns([3, 1.1], gap="medium")

with side_col:
    st.markdown('<div class="section-header">Hasil Deteksi</div>', unsafe_allow_html=True)
    col_g, col_r = st.columns(2)
    with col_g:
        good_ph = st.empty()
    with col_r:
        bad_ph = st.empty()
    prog_ph  = st.empty()
    alert_ph = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Info Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="info-table">
      <tr><td>Model</td><td><span class="accent">pills-deteks/v3</span></td></tr>
      <tr><td>Provider</td><td>Roboflow</td></tr>
      <tr><td>Threshold</td><td>0.70</td></tr>
      <tr><td>Interval</td><td>0.3 s</td></tr>
      <tr><td>Input size</td><td>224 × 224</td></tr>
    </table>""", unsafe_allow_html=True)


def _stat(v, label, cls):
    return f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value {cls}">{v}</div></div>'

def _progress(good, bad):
    total = good + bad
    pct = int(good / total * 100) if total else 0
    color = "#10d98a" if pct >= 60 else "#ef4444"
    return (f'<div class="progress-wrap"><div class="progress-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<div style="font-size:11px;font-family:\'DM Mono\',monospace;color:#4b5563;margin-top:6px;text-align:right">{pct}% pass rate</div>')

def _alert(bad):
    if bad > 0:
        return f'<div class="alert-reject"><div class="alert-dot"></div>⚠ REJECT — {bad} OBAT RUSAK</div>'
    return '<div class="alert-ok">✓ SEMUA OBAT OK</div>'


# Precompute grid overlay as a cached numpy array (drawn once, blended fast)
_GRID_CACHE: dict = {}

def _get_grid(h, w):
    key = (h, w)
    if key not in _GRID_CACHE:
        grid = np.zeros((h, w, 3), dtype=np.uint8)
        step = 32
        grid[::step, :] = (40, 42, 60)
        grid[:, ::step] = (40, 42, 60)
        _GRID_CACHE[key] = grid
    return _GRID_CACHE[key]


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_predictions = []
        self.last_request     = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = img.shape[:2]

        # API call every 0.3s — non-blocking for other frames
        now = time.time()
        if now - self.last_request > 0.3:
            self.last_request = now
            small = cv2.resize(img, (224, 224))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                try:
                    res = requests.post(
                        ROBOFLOW_URL,
                        files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                        timeout=5
                    )
                    self.last_predictions = res.json().get("predictions", [])
                except Exception:
                    pass

        scale_x = orig_w / 224
        scale_y = orig_h / 224

        # Grid overlay — precomputed, single addWeighted (fast)
        grid = _get_grid(orig_h, orig_w)
        cv2.addWeighted(grid, 0.12, img, 0.88, 0, img)

        # Corner brackets
        L = 22
        bc = (99, 102, 241)
        cv2.line(img, (0, 0),           (L, 0),           bc, 2)
        cv2.line(img, (0, 0),           (0, L),           bc, 2)
        cv2.line(img, (orig_w-L, 0),    (orig_w, 0),      bc, 2)
        cv2.line(img, (orig_w, 0),      (orig_w, L),      bc, 2)
        cv2.line(img, (0, orig_h-L),    (0, orig_h),      bc, 2)
        cv2.line(img, (0, orig_h),      (L, orig_h),      bc, 2)
        cv2.line(img, (orig_w-L, orig_h),(orig_w, orig_h), bc, 2)
        cv2.line(img, (orig_w, orig_h-L),(orig_w, orig_h), bc, 2)

        good_count = 0
        bad_count  = 0

        for pred in self.last_predictions:
            conf = pred.get("confidence", 0)
            if conf < 0.70:
                continue

            kelas = pred["class"]
            x  = int(pred["x"] * scale_x)
            y  = int(pred["y"] * scale_y)
            w  = int(pred["width"]  * scale_x)
            h  = int(pred["height"] * scale_y)
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            if kelas == "obat_bagus":
                color = (138, 217, 16)  # BGR teal-green
                good_count += 1
                label = f"obat_bagus {conf:.2f}"
            elif kelas == "obat_rusak":
                color = (68, 68, 239)   # BGR red
                bad_count += 1
                label = f"obat_rusak {conf:.2f}"
            else:
                color = (60, 200, 255)
                label = f"{kelas} {conf:.2f}"

            # Bounding box with subtle inner fill
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label badge
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            by1 = max(y1 - th - 8, 0)
            by2 = max(y1, th + 4)
            bx2 = min(x1 + tw + 10, orig_w)
            badge = img.copy()
            cv2.rectangle(badge, (x1, by1), (bx2, by2), color, -1)
            cv2.addWeighted(badge, 0.85, img, 0.15, 0, img)
            cv2.putText(img, label, (x1 + 4, by2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

        # HUD counters (top-left)
        for txt, col, yp in [
            (f"BAGUS  {good_count}", (138, 217, 16), 28),
            (f"RUSAK  {bad_count}",  (68,  68, 239), 58),
        ]:
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            hud = img.copy()
            cv2.rectangle(hud, (6, yp - th - 5), (6 + tw + 12, yp + 6), (10, 12, 18), -1)
            cv2.addWeighted(hud, 0.72, img, 0.28, 0, img)
            cv2.putText(img, txt, (12, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        # REJECT overlay
        if bad_count > 0:
            ov = img.copy()
            cv2.rectangle(ov, (0, 0), (orig_w, orig_h), (68, 68, 239), -1)
            cv2.addWeighted(ov, 0.08, img, 0.92, 0, img)
            cv2.rectangle(img, (0, 0), (orig_w, orig_h), (68, 68, 239), 3)

            rtxt = "REJECT"
            (rw, rh), _ = cv2.getTextSize(rtxt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            rx = (orig_w - rw) // 2
            ry = orig_h - 20
            rb = img.copy()
            cv2.rectangle(rb, (rx - 12, ry - rh - 8), (rx + rw + 12, ry + 8), (50, 50, 200), -1)
            cv2.addWeighted(rb, 0.85, img, 0.15, 0, img)
            cv2.putText(img, rtxt, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Camera — rendered once, never re-entered
with cam_col:
    ctx = webrtc_streamer(
        key="deteksi-obat",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"facingMode": {"ideal": "environment"}},
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

# Sidebar stats — read from processor without triggering rerun
if ctx.video_processor:
    vp   = ctx.video_processor
    good = vp.last_predictions and sum(
        1 for p in vp.last_predictions if p.get("class") == "obat_bagus" and p.get("confidence", 0) >= 0.70
    )
    bad  = vp.last_predictions and sum(
        1 for p in vp.last_predictions if p.get("class") == "obat_rusak" and p.get("confidence", 0) >= 0.70
    )
    good = good or 0
    bad  = bad  or 0
else:
    good, bad = 0, 0

good_ph.markdown(_stat(good, "BAGUS", "stat-good"), unsafe_allow_html=True)
bad_ph.markdown(_stat(bad,   "RUSAK", "stat-bad"),  unsafe_allow_html=True)
prog_ph.markdown(_progress(good, bad), unsafe_allow_html=True)
alert_ph.markdown(
    _alert(bad) if ctx.video_processor else '<div class="alert-ok" style="opacity:.5">Menunggu kamera…</div>',
    unsafe_allow_html=True
)