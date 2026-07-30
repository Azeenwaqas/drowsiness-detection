"""
Streamlit Dashboard — DrowsyGuard
CoderAxo Internship CAX-OL-2026-290
Author: Muhammad Azeen Waqas
Institution: COMSATS University Islamabad, Wah Campus
Run: cd Source_Code && py -3.11 -m streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import cv2, time

st.set_page_config(page_title="DrowsyGuard", page_icon="🚗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}
.stApp {
    background: radial-gradient(circle at 10% 20%, rgb(240, 244, 248) 0%, rgb(220, 226, 235) 90.2%);
}

[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="metric-container"] label {
    color: #475569 !important;
}
[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #0f172a !important;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 40px 0 rgba(31, 38, 135, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 1.0) !important;
}

.alert-ok {
    background: linear-gradient(135deg, rgba(200, 255, 220, 0.9), rgba(240, 255, 245, 0.9));
    border-left: 5px solid #10b981;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 17px;
    font-weight: 600;
    color: #065f46;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.15);
}

.alert-warn {
    background: linear-gradient(135deg, rgba(254, 240, 138, 0.9), rgba(254, 252, 216, 0.9));
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 17px;
    font-weight: 600;
    color: #92400e;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.15);
}

.alert-danger {
    background: linear-gradient(135deg, rgba(254, 202, 202, 0.9), rgba(254, 226, 226, 0.9));
    border-left: 5px solid #ef4444;
    border-radius: 12px;
    padding: 20px 22px;
    font-size: 20px;
    font-weight: 700;
    color: #991b1b;
    margin: 10px 0;
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);
    animation: pulse-red 1.5s infinite;
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.header-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 20px 25px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.9);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
}

.title-text {
    margin: 0;
    background: linear-gradient(to right, #3b82f6 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
  <h2 class="title-text">🚗 DrowsyGuard — Real-Time Drowsiness Monitor</h2>
  <p style="margin:8px 0 0;color:#64748b;font-size:14px;letter-spacing:0.5px;">
    CoderAxo AI/ML Internship &nbsp;•&nbsp; CAX-OL-2026-290 &nbsp;•&nbsp;
    <b style="color:#334155;">Muhammad Azeen Waqas</b> &nbsp;•&nbsp;
    COMSATS University Islamabad, Wah Campus &nbsp;•&nbsp;
    <span style="color:#94a3b8;font-weight:600;">YawDD Dataset Method + SVM Model</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Session state for run/stop ──
if 'running' not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([2, 1])
with col1:
    frame_ph = st.empty()
with col2:
    state_ph = st.empty()
    st.write("")
    rc1, rc2   = st.columns(2)
    ear_ph     = rc1.empty()
    mar_ph     = rc2.empty()
    st.write("")
    rc3, rc4   = st.columns(2)
    perclos_ph = rc3.empty()
    yawn_ph    = rc4.empty()
    st.write("")
    alert_ph   = st.empty()
    st.write("")
    rc5, rc6   = st.columns(2)
    dur_ph     = rc5.empty()
    ml_ph      = rc6.empty()

bc1, bc2 = st.columns(2)
start = bc1.button("▶ Start Monitoring", type="primary", use_container_width=True)
stop  = bc2.button("⏹ Stop",                             use_container_width=True)

if start:
    st.session_state.running = True
    st.rerun()
if stop:
    st.session_state.running = False
    st.rerun()

if st.session_state.running:
    from detector import DrowsinessDetector
    cap = cv2.VideoCapture(0)
    det = DrowsinessDetector()

    if not cap.isOpened():
        st.error("❌ Webcam not found! Close any other app using it.")
        st.session_state.running = False
    else:
        # Run until Stop pressed or webcam fails
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Webcam frame lost.")
                break

            frame, data = det.process_frame(frame)

            # ── Video feed ──
            frame_ph.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width=640          # fixed width — no deprecation warning
            )

            state     = data['state']
            alert_msg = data.get('alert_msg', '')

            # ── State badge ──
            icons = {"ALERT":"🟢","DROWSY":"🟡","VERY_DROWSY":"🔴","AWAY":"🔵","NO_FACE":"⚫"}
            icon  = icons.get(state, "⚪")
            state_ph.markdown(
                f"<h2 style='margin:0;color:#1e293b;'>{icon} {state.replace('_',' ')}</h2>",
                unsafe_allow_html=True)

            # ── Metrics ──
            ear_ph.metric("👁 EYES",       f"{data['eyes']:.3f}")
            mar_ph.metric("👄 MOUTH (MAR)", f"{data['mar']:.3f}")
            perclos_ph.metric("📊 PERCLOS", f"{data.get('perclos',0):.0%}")
            yawn_ph.metric("😮 Yawns/min", data.get('yawn_count', 0))
            dur_ph.metric("⏱ Session",    f"{data['session_duration']}s")
            ml_ph.metric("🤖 ML",
                f"{data.get('ml_label','—')} {data.get('confidence',0)*100:.0f}%")

            # ── Alert message box ──
            if state == "VERY_DROWSY":
                alert_ph.markdown(
                    f'<div class="alert-danger">🚨 {alert_msg}</div>',
                    unsafe_allow_html=True)
            elif state in ("DROWSY", "NO_FACE", "AWAY") and alert_msg:
                alert_ph.markdown(
                    f'<div class="alert-warn">⚠️ {alert_msg}</div>',
                    unsafe_allow_html=True)
            else:
                alert_ph.markdown(
                    '<div class="alert-ok">✅ Driver Alert — All Good</div>',
                    unsafe_allow_html=True)

            time.sleep(0.03)

        cap.release()
        det._sound("ALERT")
        st.session_state.running = False
        st.info("✅ Monitoring stopped.")