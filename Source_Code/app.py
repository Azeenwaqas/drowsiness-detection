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
    background: radial-gradient(circle at 10% 20%, rgb(18, 18, 28) 0%, rgb(8, 8, 16) 90.2%);
}

[data-testid="metric-container"] {
    background: rgba(30, 30, 50, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}

.alert-ok {
    background: linear-gradient(135deg, rgba(10, 46, 24, 0.8), rgba(0, 204, 68, 0.2));
    border-left: 5px solid #00ff66;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 17px;
    font-weight: 600;
    color: #a0ffc4;
    margin: 10px 0;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 15px rgba(0, 255, 102, 0.1);
}

.alert-warn {
    background: linear-gradient(135deg, rgba(46, 26, 0, 0.8), rgba(255, 170, 0, 0.2));
    border-left: 5px solid #ffcc00;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 17px;
    font-weight: 600;
    color: #ffe680;
    margin: 10px 0;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 15px rgba(255, 204, 0, 0.1);
}

.alert-danger {
    background: linear-gradient(135deg, rgba(60, 0, 0, 0.9), rgba(255, 34, 34, 0.3));
    border-left: 5px solid #ff3333;
    border-radius: 12px;
    padding: 20px 22px;
    font-size: 20px;
    font-weight: 700;
    color: #ffb3b3;
    margin: 10px 0;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(255, 51, 51, 0.2);
    animation: pulse-red 1.5s infinite;
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(255, 51, 51, 0.4); }
    70% { box-shadow: 0 0 0 15px rgba(255, 51, 51, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 51, 51, 0); }
}

.header-card {
    background: rgba(20, 20, 35, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 20px 25px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.title-text {
    margin: 0;
    background: linear-gradient(to right, #00f2fe 0%, #4facfe 100%);
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
  <p style="margin:8px 0 0;color:#9ca3af;font-size:14px;letter-spacing:0.5px;">
    CoderAxo AI/ML Internship &nbsp;•&nbsp; CAX-OL-2026-290 &nbsp;•&nbsp;
    <b style="color:#f3f4f6;">Muhammad Azeen Waqas</b> &nbsp;•&nbsp;
    COMSATS University Islamabad, Wah Campus &nbsp;•&nbsp;
    <span style="color:#6b7280;font-weight:600;">YawDD Dataset Method + SVM Model</span>
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
if stop:
    st.session_state.running = False

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
            icons = {"ALERT":"🟢","DROWSY":"🟡","VERY_DROWSY":"🔴","NO_FACE":"⚫"}
            icon  = icons.get(state, "⚪")
            state_ph.markdown(
                f"<h2 style='margin:0;color:#ddd;'>{icon} {state.replace('_',' ')}</h2>",
                unsafe_allow_html=True)

            # ── Metrics ──
            ear_ph.metric("👁 EAR",       f"{data['ear']:.3f}")
            mar_ph.metric("👄 MAR",       f"{data['mar']:.3f}")
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
            elif state in ("DROWSY", "NO_FACE") and alert_msg:
                alert_ph.markdown(
                    f'<div class="alert-warn">⚠️ {alert_msg}</div>',
                    unsafe_allow_html=True)
            else:
                alert_ph.markdown(
                    '<div class="alert-ok">✅ Driver Alert — All Good</div>',
                    unsafe_allow_html=True)

            time.sleep(0.03)

        cap.release()
        st.session_state.running = False
        st.info("✅ Monitoring stopped.")