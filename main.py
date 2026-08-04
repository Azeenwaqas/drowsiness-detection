import sys, os
import cv2
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from detector import DrowsinessDetector

cap = None
det = DrowsinessDetector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cap
    import platform
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    yield
    if cap is not None:
        cap.release()

app = FastAPI(title="DrowsyGuard", description="Real-Time Drowsiness Monitor Backend", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

is_running = False
latest_frame = None
latest_data = {
    'state': 'STOPPED',
    'alert_msg': 'Monitoring Stopped',
    'eyes': 0.0,
    'mar': 0.0,
    'perclos': 0.0,
    'yawn_count': 0,
    'session_duration': 0,
    'ml_label': '-',
    'confidence': 0.0
}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def generate_frames():
    global is_running, latest_frame, latest_data, cap
    
    # Create a blank black frame to yield when stopped or failed
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ret, buffer = cv2.imencode('.jpg', blank_frame)
    blank_bytes = buffer.tobytes()

    while True:
        if not is_running:
            # Yield black frame so stream does not hang and browser can connect
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
            time.sleep(0.1)
            continue
            
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue
            
        success, frame = cap.read()
        if not success:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
            time.sleep(0.1)
            continue
            
        frame, data = det.process_frame(frame)
        latest_frame = frame
        latest_data = data
        
        # Encode for web stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global is_running, latest_data
    try:
        while True:
            # send data every 100ms
            if is_running:
                await websocket.send_json(latest_data)
            else:
                await websocket.send_json({
                    "state": "STOPPED",
                    "alert_msg": "System is currently stopped.",
                    "eyes": 0, "mar": 0, "perclos": 0, "yawn_count": 0,
                    "session_duration": 0, "ml_label": "-", "confidence": 0
                })
            
            # check for incoming messages (like start/stop) non-blocking
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                if msg == "start":
                    is_running = True
                elif msg == "stop":
                    is_running = False
                    det._sound("ALERT") # Stop alarm if running
            except asyncio.TimeoutError:
                pass
                
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
