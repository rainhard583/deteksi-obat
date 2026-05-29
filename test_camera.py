import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="Test Kamera Realtime")

st.title("TEST KAMERA REALTIME")

st.write("Jika berhasil, akan muncul tombol START dan video realtime.")

webrtc_streamer(
    key="kamera-test",
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)