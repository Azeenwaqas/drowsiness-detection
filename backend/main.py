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
from fastapi.middleware.cors import CORSMiddleware

cap = None
det = DrowsinessDetector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="DrowsyGuard", description="Real-Time Drowsiness Monitor Backend", lifespan=lifespan)

# Add CORS Middleware for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (update with Vercel URL in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"message": "DrowsyGuard Backend API is running."}

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

# Pure API Backend - No HTML rendering needed

import base64

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global is_running, latest_data
    try:
        while True:
            # We expect the client to send a JSON with {"type": "...", "image": "..."}
            msg = await websocket.receive_json()
            
            if msg.get("type") == "start":
                is_running = True
            elif msg.get("type") == "stop":
                is_running = False
                det._sound("ALERT") # Stop alarm
                await websocket.send_json({
                    "data": {
                        "state": "STOPPED",
                        "alert_msg": "System is currently stopped.",
                        "eyes": 0, "mar": 0, "perclos": 0, "yawn_count": 0,
                        "session_duration": 0, "ml_label": "-", "confidence": 0
                    }
                })
            elif msg.get("type") == "frame" and is_running:
                # Decode base64 frame from client
                encoded_data = msg["image"].split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Process frame
                    out_frame, data = det.process_frame(img)
                    
                    # Encode output frame back to base64
                    ret, buffer = cv2.imencode('.jpg', out_frame)
                    if ret:
                        out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                        
                        # Send back processed frame and data
                        await websocket.send_json({
                            "image": out_b64,
                            "data": data
                        })
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
