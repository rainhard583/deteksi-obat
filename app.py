import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import requests
import time
import numpy as np

API_KEY = "uc5sGP3TXol5G8dbrmo3"

ROBOFLOW_URL = (
    f"https://detect.roboflow.com/pills-deteks/3?api_key={API_KEY}"
)

st.set_page_config(
    page_title="Deteksi Obat AI",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: #0a0c10;
    color: #e8eaf0;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: #111318;
    border: 1px solid #1f2230;
    border-radius: 12px;
    margin-bottom: 20px;
}
.app-title {
    font-size: 18px;
    font-weight: 600;
    color: #e8eaf0;
    letter-spacing: -0.3px;
}
.app-subtitle {
    font-size: 12px;
    color: #4b5563;
    font-family: 'DM Mono', monospace;
    margin-top: 2px;
}
.live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    color: #10d98a;
    background: rgba(16,217,138,0.08);
    border: 1px solid rgba(16,217,138,0.2);
    border-radius: 20px;
    padding: 5px 12px;
}
.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10d98a;
    animation: pulse 1.4s ease-in-out infinite;
    display: inline-block;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}

/* ── Stat cards ── */
.stat-card {
    background: #111318;
    border: 1px solid #1f2230;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.stat-label {
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 8px;
}
.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 42px;
    font-weight: 500;
    line-height: 1;
}
.stat-good { color: #10d98a; }
.stat-bad  { color: #ef4444; }

/* ── Alert ── */
.alert-reject {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    letter-spacing: 1px;
    color: #ef4444;
    margin-top: 12px;
    animation: alertBlink 0.8s ease-in-out infinite;
}
@keyframes alertBlink {
    0%, 100% { border-color: rgba(239,68,68,0.35); }
    50% { border-color: rgba(239,68,68,0.7); }
}
.alert-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #ef4444;
    animation: pulse 0.8s ease-in-out infinite;
    flex-shrink: 0;
}

/* ── Clear alert ── */
.alert-ok {
    background: rgba(16, 217, 138, 0.06);
    border: 1px solid rgba(16, 217, 138, 0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #10d98a;
    margin-top: 12px;
}

/* ── Info table ── */
.info-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
.info-table td {
    padding: 6px 0;
    border-bottom: 1px solid #1a1d28;
}
.info-table td:first-child {
    color: #4b5563;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.info-table td:last-child {
    color: #9ca3af;
    text-align: right;
    font-family: 'DM Mono', monospace;
}
.info-table span.accent { color: #6366f1; }

/* ── Progress bar ── */
.progress-wrap {
    background: #1a1d28;
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
    margin-top: 12px;
}
.progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* ── Section header ── */
.section-header {
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 12px;
    margin-top: 4px;
}

/* ── Streamlit overrides ── */
div[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; }
div[data-testid="column"] > div { height: 100%; }
.stButton > button {
    background: transparent;
    border: 1px solid #1f2230;
    color: #9ca3af;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: #6366f1;
    color: #e8eaf0;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <div class="app-title">💊 Deteksi Obat AI</div>
    <div class="app-subtitle">pills-deteks/v3 · Roboflow · confidence ≥ 0.70</div>
  </div>
  <div class="live-badge">
    <span class="live-dot"></span>
    REALTIME
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ───────────────────────────────────────────────────────────────────
cam_col, side_col = st.columns([3, 1.1], gap="medium")

# ─── Sidebar placeholders ──────────────────────────────────────────────────────
with side_col:
    st.markdown('<div class="section-header">Hasil Deteksi</div>', unsafe_allow_html=True)

    col_g, col_r = st.columns(2)
    with col_g:
        good_placeholder = st.empty()
    with col_r:
        bad_placeholder = st.empty()

    progress_placeholder = st.empty()
    alert_placeholder    = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Info Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="info-table">
      <tr>
        <td>Model</td>
        <td><span class="accent">pills-deteks/v3</span></td>
      </tr>
      <tr>
        <td>Provider</td>
        <td>Roboflow</td>
      </tr>
      <tr>
        <td>Threshold</td>
        <td>0.70</td>
      </tr>
      <tr>
        <td>Interval</td>
        <td>0.3 s</td>
      </tr>
      <tr>
        <td>Input size</td>
        <td>224 × 224</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "good_count" not in st.session_state:
    st.session_state.good_count = 0
    st.session_state.bad_count  = 0


def _render_stat(value: int, label: str, color_class: str) -> str:
    return f"""
    <div class="stat-card">
      <div class="stat-label">{label}</div>
      <div class="stat-value {color_class}">{value}</div>
    </div>"""


def _render_progress(good: int, bad: int) -> str:
    total = good + bad
    pct   = int(good / total * 100) if total else 0
    color = "#10d98a" if pct >= 60 else "#ef4444"
    return f"""
    <div class="progress-wrap">
      <div class="progress-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <div style="font-size:11px; font-family:'DM Mono',monospace;
                color:#4b5563; margin-top:6px; text-align:right;">
      {pct}% pass rate
    </div>"""


def _render_alert(bad: int) -> str:
    if bad > 0:
        return f"""
        <div class="alert-reject">
          <div class="alert-dot"></div>
          ⚠ REJECT — {bad} OBAT RUSAK
        </div>"""
    return '<div class="alert-ok">✓ SEMUA OBAT OK</div>'


# ─── Video Processor ──────────────────────────────────────────────────────────
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_predictions = []
        self.last_request     = 0
        self.good_count       = 0
        self.bad_count        = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = img.shape[:2]

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

        good_count = 0
        bad_count  = 0

        # ── subtle grid overlay ──────────────────────────────────────────────
        overlay_grid = img.copy()
        step_px = 32
        for gx in range(0, orig_w, step_px):
            cv2.line(overlay_grid, (gx, 0), (gx, orig_h), (99, 102, 241), 1)
        for gy in range(0, orig_h, step_px):
            cv2.line(overlay_grid, (0, gy), (orig_w, gy), (99, 102, 241), 1)
        cv2.addWeighted(overlay_grid, 0.04, img, 0.96, 0, img)

        # ── corner brackets ──────────────────────────────────────────────────
        L, T_c = 22, 22
        col_br = (99, 102, 241)
        for (x0, y0, dx, dy) in [
            (0, 0, L, 0), (0, 0, 0, T_c),
            (orig_w-L, 0, orig_w, 0), (orig_w, 0, orig_w, T_c),
            (0, orig_h-T_c, 0, orig_h), (0, orig_h, L, orig_h),
            (orig_w-L, orig_h, orig_w, orig_h), (orig_w, orig_h-T_c, orig_w, orig_h),
        ]:
            cv2.line(img, (x0, y0), (dx, dy), col_br, 2)

        # ── predictions ──────────────────────────────────────────────────────
        for pred in self.last_predictions:
            conf = pred.get("confidence", 0)
            if conf < 0.70:
                continue

            kelas = pred["class"]
            x     = int(pred["x"] * scale_x)
            y     = int(pred["y"] * scale_y)
            w     = int(pred["width"]  * scale_x)
            h     = int(pred["height"] * scale_y)
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            if kelas == "obat_bagus":
                color = (16, 217, 138)   # teal-green
                good_count += 1
                label = f"obat_bagus  {conf:.2f}"
            elif kelas == "obat_rusak":
                color = (68, 68, 239)    # red (BGR)
                bad_count += 1
                label = f"obat_rusak  {conf:.2f}"
            else:
                color = (255, 200, 60)
                label = f"{kelas}  {conf:.2f}"

            # glow rect via layered rectangles
            for thickness, alpha in [(8, 0.06), (4, 0.12), (2, 1.0)]:
                glow = img.copy()
                cv2.rectangle(glow, (x1, y1), (x2, y2), color, thickness)
                cv2.addWeighted(glow, alpha, img, 1 - alpha, 0, img) if alpha < 1 else None
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # pill label badge
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_x1 = x1
            badge_y1 = max(y1 - th - 10, 0)
            badge_x2 = x1 + tw + 10
            badge_y2 = max(y1, th + 10)
            badge_bg = img.copy()
            cv2.rectangle(badge_bg, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.addWeighted(badge_bg, 0.85, img, 0.15, 0, img)
            cv2.putText(
                img, label,
                (badge_x1 + 5, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (10, 10, 10), 1, cv2.LINE_AA
            )

        # ── HUD counters ─────────────────────────────────────────────────────
        hud_items = [
            (f"BAGUS  {good_count}", (16, 217, 138), 28),
            (f"RUSAK  {bad_count}",  (68,  68, 239), 58),
        ]
        for (txt, col, y_pos) in hud_items:
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            hud_bg = img.copy()
            cv2.rectangle(hud_bg, (8, y_pos - th - 6), (8 + tw + 12, y_pos + 6), (10, 12, 18), -1)
            cv2.addWeighted(hud_bg, 0.75, img, 0.25, 0, img)
            cv2.putText(img, txt, (14, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        # ── REJECT state ─────────────────────────────────────────────────────
        if bad_count > 0:
            border_overlay = img.copy()
            cv2.rectangle(border_overlay, (0, 0), (orig_w, orig_h), (68, 68, 239), -1)
            cv2.addWeighted(border_overlay, 0.10, img, 0.90, 0, img)
            cv2.rectangle(img, (0, 0), (orig_w, orig_h), (68, 68, 239), 4)

            rej_txt = "REJECT"
            (rw, rh), _ = cv2.getTextSize(rej_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            rx = (orig_w - rw) // 2
            ry = orig_h - 24
            rej_bg = img.copy()
            cv2.rectangle(rej_bg, (rx - 12, ry - rh - 8), (rx + rw + 12, ry + 8), (68, 68, 239), -1)
            cv2.addWeighted(rej_bg, 0.85, img, 0.15, 0, img)
            cv2.putText(img, rej_txt, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

        self.good_count = good_count
        self.bad_count  = bad_count

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── Camera stream ────────────────────────────────────────────────────────────
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

# ─── Live stat update ─────────────────────────────────────────────────────────
if ctx.video_processor:
    vp = ctx.video_processor
    good = vp.good_count
    bad  = vp.bad_count

    good_placeholder.markdown(_render_stat(good, "BAGUS", "stat-good"), unsafe_allow_html=True)
    bad_placeholder.markdown(_render_stat(bad,  "RUSAK", "stat-bad"),  unsafe_allow_html=True)
    progress_placeholder.markdown(_render_progress(good, bad), unsafe_allow_html=True)
    alert_placeholder.markdown(_render_alert(bad), unsafe_allow_html=True)
else:
    good_placeholder.markdown(_render_stat(0, "BAGUS", "stat-good"), unsafe_allow_html=True)
    bad_placeholder.markdown(_render_stat(0, "RUSAK", "stat-bad"),  unsafe_allow_html=True)
    progress_placeholder.markdown(_render_progress(0, 0), unsafe_allow_html=True)
    alert_placeholder.markdown(
        '<div class="alert-ok" style="opacity:0.5;">Menunggu kamera…</div>',
        unsafe_allow_html=True
    )