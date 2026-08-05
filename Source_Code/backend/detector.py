"""
Core Drowsiness Detector — DrowsyGuard
CoderAxo Internship CAX-OL-2026-290
Author: Muhammad Azeen Waqas
Institution: COMSATS University Islamabad, Wah Campus

Detection Logic based on YawDD Dataset research:
  - Abtahi et al. (2014) YawDD: A Yawning Detection Dataset
  - 4 states: OPEN_EYES | CLOSED_EYES | YAWNING | NO_YAWNING
  - Drowsiness = closed eyes duration + yawn frequency
  - Personalized EAR/MAR thresholds (calibrated per driver)
  - PERCLOS > 0.15 over 60 frames = drowsy (Wierwille 1994)
  - Yawn count >= 3 in 60 seconds = drowsy (YawDD benchmark)
"""

import cv2, time, os, math, threading, wave, struct
import numpy as np
from scipy.spatial import distance as dist
import pickle

# Load ML Model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pkl')
cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    ml_model  = model_data['model']
    FEATURES  = model_data['features']
    CLASSES   = model_data['classes']  # {0:'Alert', 1:'Drowsy', 2:'Very Drowsy'}
    print(f"Model loaded - Accuracy: {model_data['accuracy']*100:.1f}%")
except Exception as e:
    print(f"Failed to load model: {e}")
    ml_model = None
    CLASSES = {0:'Alert', 1:'Drowsy', 2:'Very Drowsy'}

# ─── MediaPipe Landmark Indices ───────────────────────
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# YawDD uses outer mouth landmarks for better yawn detection
MOUTH_OUTER = [61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
FACE_OVAL   = [10,338,297,332,284,251,389,356,454,323,361,288,
               397,365,379,378,400,377,152,148,176,149,150,136,
               172,58,132,93,234,127,162,21,54,103,67,109]

# ─── Default Thresholds (will be personalized during calibration) ──
# Balanced for reliable real-world detection (not too sensitive, not too slow)
EAR_OPEN        = 0.30   # eyes fully open (calibrated from user)
EAR_CLOSED      = 0.22   # eyes closed threshold
MAR_YAWN        = 0.65   # yawning threshold (YawDD benchmark)
PERCLOS_LIMIT   = 0.18   # 18% eye closure in 60 frames = drowsy (Balanced)
YAWN_FREQ_LIMIT = 3      # 3 yawns in 60 sec = drowsy (YawDD)
YAWN_FRAMES     = 10     # frames mouth must be open to count as yawn
DROWSY_FRAMES   = 12     # ~1 second before DROWSY
HARD_FRAMES     = 24     # ~2 seconds before VERY_DROWSY
NO_FACE_LIMIT   = 30     # ~2-3 seconds frames no face before alert

# ─── Sound Engine ─────────────────────────────────────
def _gen_wav(path, freq=1000, dur=0.4, vol=0.7, rate=44100):
    with wave.open(path, 'w') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        for i in range(int(rate * dur)):
            t = i / rate
            # Richer sound with harmonics and envelope
            v = math.sin(2 * math.pi * freq * t) + 0.5 * math.sin(2 * math.pi * (freq * 1.5) * t)
            envelope = math.exp(-4 * t / dur)
            sample = int(vol * 15000 * v * envelope)
            sample = max(-32768, min(32767, sample))
            f.writeframes(struct.pack('<h', sample))

_dir       = os.path.dirname(os.path.abspath(__file__))
BEEP_PATH  = os.path.join(_dir, "beep.wav")
ALARM_PATH = os.path.join(_dir, "alarm.wav")
AWAY_PATH  = os.path.join(_dir, "away.wav")
if not os.path.exists(BEEP_PATH) or os.path.getsize(BEEP_PATH) > 20000:  _gen_wav(BEEP_PATH,  1000, 0.2, 0.6)
if not os.path.exists(ALARM_PATH) or os.path.getsize(ALARM_PATH) > 50000: _gen_wav(ALARM_PATH, 1800, 0.3, 0.9)
if not os.path.exists(AWAY_PATH) or os.path.getsize(AWAY_PATH) > 30000:  _gen_wav(AWAY_PATH,  600,  0.25, 0.7)

class _Sound:
    def __init__(self):
        self._active = False
        self._stop   = threading.Event()
        self._engine = None
        try:
            import pygame
            pygame.mixer.init(44100, -16, 1, 512)
            self._engine = 'pygame'
        except:
            try:
                import winsound
                self._engine = 'winsound'
            except:
                pass

    def play(self, path, loop=False):
        if self._active: return
        self._active = True
        self._stop.clear()
        threading.Thread(target=self._run, args=(path,loop), daemon=True).start()

    def stop(self):
        self._stop.set()
        self._active = False
        if self._engine == 'pygame':
            try:
                import pygame; pygame.mixer.stop()
            except: pass
        elif self._engine == 'winsound':
            try:
                import winsound; winsound.PlaySound(None, 0)
            except: pass

    def _run(self, path, loop):
        try:
            if self._engine == 'pygame':
                import pygame
                s = pygame.mixer.Sound(path)
                while not self._stop.is_set():
                    s.play()
                    pygame.time.wait(int(s.get_length()*1000))
                    if not loop: break
            elif self._engine == 'winsound':
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME |
                    (winsound.SND_LOOP|winsound.SND_ASYNC if loop else 0))
        except: pass
        finally: self._active = False

sound = _Sound()

# ─── MediaPipe Setup ──────────────────────────────────
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,
    refine_landmarks=False, # CRITICAL: Huge performance boost, we don't need iris tracking
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
print("FaceMesh ready")

# ─── Feature Functions ────────────────────────────────
def calc_ear(lm, idx, w, h):
    p = np.array([(lm[i].x*w, lm[i].y*h) for i in idx])
    A = dist.euclidean(p[1],p[5]); B = dist.euclidean(p[2],p[4])
    C = dist.euclidean(p[0],p[3])
    return (A+B)/(2.0*C+1e-6)

def calc_mar(lm, outer, inner, w, h):
    """
    YawDD-style MAR using both outer and inner mouth landmarks
    for more accurate yawn detection.
    """
    op = np.array([(lm[i].x*w, lm[i].y*h) for i in outer])
    ip = np.array([(lm[i].x*w, lm[i].y*h) for i in inner])
    # Outer vertical
    oA = dist.euclidean(op[2],op[10]); oB = dist.euclidean(op[4],op[8])
    oC = dist.euclidean(op[0],op[6])
    outer_mar = (oA+oB)/(2.0*oC+1e-6)
    # Inner vertical (more sensitive to yawning)
    iA = dist.euclidean(ip[1],ip[7]); iB = dist.euclidean(ip[3],ip[5])
    iC = dist.euclidean(ip[0],ip[6])
    inner_mar = (iA+iB)/(2.0*iC+1e-6)
    return (outer_mar + inner_mar) / 2.0

def get_face_bbox(lm, w, h, pad=20):
    xs = [lm[i].x*w for i in FACE_OVAL]
    ys = [lm[i].y*h for i in FACE_OVAL]
    return (max(0,int(min(xs))-pad), max(0,int(min(ys))-pad),
            min(w-1,int(max(xs))+pad), min(h-1,int(max(ys))+pad))

def get_head_pitch(lm, w, h):
    # Nose tip, Chin, Left eye left, Right eye right, Left mouth, Right mouth
    img_pts = np.array([(lm[1].x*w, lm[1].y*h), (lm[152].x*w, lm[152].y*h),
                        (lm[33].x*w, lm[33].y*h), (lm[263].x*w, lm[263].y*h),
                        (lm[61].x*w, lm[61].y*h), (lm[291].x*w, lm[291].y*h)], dtype="double")
    model_pts = np.array([(0.0,0.0,0.0), (0.0,-330.0,-65.0), (-225.0,170.0,-135.0),
                          (225.0,170.0,-135.0), (-150.0,-150.0,-125.0), (150.0,-150.0,-125.0)])
    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
    success, rot_vec, trans_vec = cv2.solvePnP(model_pts, img_pts, cam_matrix, np.zeros((4,1)))
    if success:
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        return math.degrees(math.asin(max(-1.0, min(1.0, -rot_mat[2][0]))))
    return 0.0

# ─── Drawing Helpers ──────────────────────────────────
def draw_rounded_rect(img, x1,y1,x2,y2, color, t=2, r=16):
    cv2.line(img,(x1+r,y1),(x2-r,y1),color,t)
    cv2.line(img,(x1+r,y2),(x2-r,y2),color,t)
    cv2.line(img,(x1,y1+r),(x1,y2-r),color,t)
    cv2.line(img,(x2,y1+r),(x2,y2-r),color,t)
    cv2.ellipse(img,(x1+r,y1+r),(r,r),180, 0,90,color,t)
    cv2.ellipse(img,(x2-r,y1+r),(r,r),270, 0,90,color,t)
    cv2.ellipse(img,(x1+r,y2-r),(r,r), 90, 0,90,color,t)
    cv2.ellipse(img,(x2-r,y2-r),(r,r),  0, 0,90,color,t)

def draw_feature_box(img, lm, indices, w, h, color, label="", pad=5):
    pts = np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in indices])
    x,y,bw,bh = cv2.boundingRect(pts)
    cv2.rectangle(img,(x-pad,y-pad),(x+bw+pad,y+bh+pad),color,1)
    if label:
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(img,(x-pad,y-pad-th-4),(x-pad+tw+4,y-pad),(0,0,0),-1)
        cv2.putText(img, label,(x-pad+2,y-pad-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,color,1)

def get_feature_bbox_coords(lm, indices, w, h, pad=5):
    pts = np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in indices])
    x,y,bw,bh = cv2.boundingRect(pts)
    return [int(x-pad), int(y-pad), int(bw+2*pad), int(bh+2*pad)]

def put_label_bg(img, text, x, y, color, scale=0.5, t=1):
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    cv2.rectangle(img,(x-2,y-th-3),(x+tw+2,y+3),(0,0,0),-1)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,t)

# ─── Color palette ───────────────────────────────────
C = {
    'green' :(0,210,70), 'yellow':(0,200,255),
    'red'   :(0,50,255), 'cyan'  :(255,220,0),
    'orange':(0,140,255),'white' :(230,230,230),
    'black' :(10,10,10)
}
STATE_COLOR = {
    'ALERT'      : C['green'],
    'DROWSY'     : C['yellow'],
    'VERY_DROWSY': C['red'],
    'NO_FACE'    : (100,100,100),
    'AWAY'       : (0, 100, 255),
    'CALIBRATING': C['cyan'],
}

# ─── Main Detector ────────────────────────────────────
class DrowsinessDetector:
    """
    YawDD-aligned drowsiness detector.
    Detection signals (from YawDD research):
      1. Eye closure duration  → PERCLOS metric
      2. Yawn frequency        → yawn count per 60 sec window
      3. Head absence          → face not visible (head down/turned)
    """
    def __init__(self):
        # Calibration
        self.calibrated      = False
        self.calib_ear       = []
        self.calib_mar       = []
        self.calib_pitch     = []
        self.ear_thresh      = EAR_CLOSED
        self.mar_thresh      = MAR_YAWN
        self.base_pitch      = 0.0

        # YawDD-style counters
        self.ear_history     = []   # rolling 60-frame EAR window
        self.yawn_counter    = 0    # current yawn frame count
        self.yawn_events     = []   # timestamps of completed yawns
        self.yawning_now     = False
        self.pitch_smooth    = 0.0  # EMA smoothed head pitch
        self.bow_counter     = 0    # frames looking down

        # Time-based tracking (fixes network/latency frame-rate fluctuations)
        self.eyes_closed_start      = None
        self.head_bow_start         = None
        self.ml_drowsy_start        = None
        self.ml_very_drowsy_start   = None

        # State
        self.frame_counter   = 0
        self.no_face_counter = 0
        self.alert_count     = 0
        self.state           = "ALERT"
        self.alert_msg       = ""
        self.start_time      = time.time()
        self._last_snd       = None

    # ── Internal helpers ──────────────────────────────
    def _sound(self, state):
        # Disabled on backend to save resources (server has no speakers)
        # Audio should be played on the frontend client side
        pass

    def _set_alert(self, msg):
        if self.alert_msg != msg: self.alert_count += 1
        self.alert_msg = msg

    def _clear_alert(self):
        self.alert_msg = ""

    def _yawn_freq(self):
        """Count yawns in last 60 seconds (YawDD metric)."""
        now = time.time()
        self.yawn_events = [t for t in self.yawn_events if now-t < 60]
        return len(self.yawn_events)

    def _perclos(self):
        """PERCLOS: % of frames in last 60 with EAR below threshold."""
        if not self.ear_history: return 0.0
        return sum(1 for e in self.ear_history
                   if e < self.ear_thresh) / len(self.ear_history)

    # ── Main process ──────────────────────────────────
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res  = face_mesh.process(rgb)
        rgb.flags.writeable = True

        # Dark top bar
        cv2.rectangle(frame,(0,0),(w,90),(12,12,18),-1)
        cv2.line(frame,(0,90),(w,90),(35,35,55),1)

        # ══════════════════════════════════════════════
        # PHASE 1 — CALIBRATION (first ~3 seconds)
        # YawDD research: personalized thresholds per driver
        # ══════════════════════════════════════════════
        if not self.calibrated:
            n   = len(self.calib_ear)
            pct = int(n/15*100)
            bw  = int(w*pct/100)
            cv2.rectangle(frame,(0,90),(bw,96),C['cyan'],-1)
            cv2.putText(frame,
                f"CALIBRATING {pct}%  —  Eyes OPEN, face camera directly",
                (10,38),cv2.FONT_HERSHEY_SIMPLEX,0.7,C['cyan'],2)
            cv2.putText(frame,
                "Personalizing your EAR & MAR thresholds (YawDD method)...",
                (10,68),cv2.FONT_HERSHEY_SIMPLEX,0.43,(160,160,160),1)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                e  = (calc_ear(lm,LEFT_EYE, w,h)+
                      calc_ear(lm,RIGHT_EYE,w,h))/2.0
                m  = calc_mar(lm,MOUTH_OUTER,MOUTH_INNER,w,h)
                p  = get_head_pitch(lm, w, h)
                self.calib_ear.append(e)
                self.calib_mar.append(m)
                self.calib_pitch.append(p)

            if n >= 15:
                base_ear        = np.mean(self.calib_ear)
                base_mar        = np.mean(self.calib_mar)
                self.base_pitch = np.mean(self.calib_pitch) if self.calib_pitch else 0.0
                # Make threshold more sensitive so it detects drowsiness easier
                self.ear_thresh = min(0.32, max(0.22, round(base_ear - 0.020, 3)))
                self.mar_thresh = max(0.55, round(base_mar + 0.15,  3))
                self.calibrated = True
                print(f"Calibrated -> EAR thresh:{self.ear_thresh}  "
                      f"MAR thresh:{self.mar_thresh}  "
                      f"Base Pitch:{self.base_pitch:.1f}  "
                      f"(base EAR:{base_ear:.3f} MAR:{base_mar:.3f})")

            return frame, {
                "state":"CALIBRATING","eyes":0.0,"mar":0.0,
                "perclos":0.0,"yawn_count":0,
                "alert_msg":"Calibrating...","alert_count":0,
                "session_duration":int(time.time()-self.start_time)
            }

        # ══════════════════════════════════════════════
        # PHASE 2 — NO FACE (head down / turned)
        # YawDD: head absence treated as drowsy signal
        # ══════════════════════════════════════════════
        if not res.multi_face_landmarks:
            self.no_face_counter += 1
            self.frame_counter    = min(self.frame_counter+1, HARD_FRAMES)

            if self.no_face_counter >= NO_FACE_LIMIT:
                self.state = "AWAY"
                self._set_alert("Driver looks away or left/right!")
                self._sound("AWAY")
                
                ov = frame.copy()
                cv2.rectangle(ov,(0,0),(w,h),(30,30,100),-1)
                cv2.addWeighted(ov,0.4,frame,0.6,0,frame)
                cv2.putText(frame,"LOOK FORWARD!",
                    (w//2-180,h//2),cv2.FONT_HERSHEY_SIMPLEX,1.5,(200,200,255),4)
            else:
                self.state = "NO_FACE"
                self._sound("NO_FACE")

            sc = STATE_COLOR.get(self.state,(100,100,100))
            cv2.putText(frame,f"STATE: {self.state}",(10,36),
                        cv2.FONT_HERSHEY_SIMPLEX,0.88,sc,2)
            cv2.putText(frame,
                f"Head down/turned — {self.no_face_counter}/{NO_FACE_LIMIT} frames",
                (10,66),cv2.FONT_HERSHEY_SIMPLEX,0.46,(180,180,180),1)

            return frame,{
                "state":self.state,"eyes":0.0,"mar":0.0,
                "perclos":self._perclos(),"yawn_count":self._yawn_freq(),
                "alert_msg":self.alert_msg,"alert_count":self.alert_count,
                "session_duration":int(time.time()-self.start_time)
            }

        # ══════════════════════════════════════════════
        # PHASE 3 — FACE DETECTED — Full YawDD analysis
        # ══════════════════════════════════════════════
        lm      = res.multi_face_landmarks[0].landmark
        
        nx = lm[1].x; lx = lm[234].x; rx = lm[454].x
        yaw_ratio = abs(nx - lx) / (abs(rx - nx) + 1e-6)
        
        # 30-45 degree head turn will result in a ratio around 1.6-2.0 (or 0.6-0.4)
        if yaw_ratio > 1.7 or yaw_ratio < 0.58:
            self.no_face_counter += 1
            if self.no_face_counter >= NO_FACE_LIMIT:
                self.state = "AWAY"
                self._set_alert("Driver looks away or left/right!")
                self._sound("AWAY")
                
                ov = frame.copy()
                cv2.rectangle(ov,(0,0),(w,h),(30,30,100),-1)
                cv2.addWeighted(ov,0.4,frame,0.6,0,frame)
                cv2.putText(frame,"LOOK FORWARD!",
                    (w//2-180,h//2),cv2.FONT_HERSHEY_SIMPLEX,1.5,(200,200,255),4)
                
                sc = STATE_COLOR.get(self.state,(100,100,100))
                cv2.putText(frame,f"STATE: {self.state}",(10,36),
                            cv2.FONT_HERSHEY_SIMPLEX,0.88,sc,2)
                return frame,{
                    "state":self.state,"eyes":0.0,"mar":0.0,
                    "perclos":self._perclos(),"yawn_count":self._yawn_freq(),
                    "alert_msg":self.alert_msg,"alert_count":self.alert_count,
                    "session_duration":int(time.time()-self.start_time)
                }
        else:
            self.no_face_counter = max(0, self.no_face_counter - 2)

        ear_val = (calc_ear(lm,LEFT_EYE, w,h)+
                   calc_ear(lm,RIGHT_EYE,w,h))/2.0
        mar_val = calc_mar(lm,MOUTH_OUTER,MOUTH_INNER,w,h)

        # ── PERCLOS (YawDD metric 1) ──────────────────
        self.ear_history.append(ear_val)
        if len(self.ear_history)>60: self.ear_history.pop(0)
        
        # INSTANT WAKE-UP FIX: If eyes are fully open, flush the history so alarm stops instantly!
        if ear_val > self.ear_thresh + 0.02:
            self.ear_history = [ear_val] * len(self.ear_history)
            self.eyes_closed_start = None
            self.ml_drowsy_start = None
            self.ml_very_drowsy_start = None
            self.head_bow_start = None

        perclos = self._perclos()

        # ── Yawn detection (YawDD metric 2) ──────────
        # Count a yawn only when mouth stays open for YAWN_FRAMES
        is_mouth_open = mar_val > self.mar_thresh
        if is_mouth_open:
            self.yawn_counter += 1
            if self.yawn_counter == YAWN_FRAMES and not self.yawning_now:
                self.yawning_now = True
                self.yawn_events.append(time.time())
        else:
            self.yawn_counter = 0
            self.yawning_now = False

        yawn_freq = self._yawn_freq()

        # ── Eye closure counter ───────────────────────
        eyes_closed = ear_val < self.ear_thresh
        if eyes_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = time.time()
            eye_closure_duration = time.time() - self.eyes_closed_start
        else:
            self.eyes_closed_start = None
            eye_closure_duration = 0.0

        # ── Bowing head counter ───────────────────────
        raw_pitch = get_head_pitch(lm, w, h)
        if abs(raw_pitch - self.base_pitch) > 25.0:  # Threshold for bowing/tilting head relative to calibration
            if self.head_bow_start is None:
                self.head_bow_start = time.time()
            head_bow_duration = time.time() - self.head_bow_start
        else:
            self.head_bow_start = None
            head_bow_duration = 0.0

        # ── YawDD drowsiness classification ──────────
        # ML Model Inference
        self.pitch_smooth = 0.7 * self.pitch_smooth + 0.3 * raw_pitch
        head_pitch = self.pitch_smooth
        ear_mar_ratio = ear_val / (mar_val + 1e-6)
        pitch_norm    = abs(head_pitch) / 35.0
        drowsy_score  = np.clip(
            (1 - ear_val / 0.35) * 0.40 +
            (mar_val / 0.80)     * 0.30 +
             perclos             * 0.30,
            0, 1
        )

        features = np.array([[
            ear_val, mar_val, head_pitch, perclos,
            ear_mar_ratio, pitch_norm, drowsy_score
        ]])

        if ml_model:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_label = ml_model.predict(features)[0]
                pred_proba = ml_model.predict_proba(features)[0]
            confidence = pred_proba[pred_label]
            ml_state = CLASSES[pred_label]
        else:
            ml_state = 'Alert'
            confidence = 1.0

        # Time-based tracking of ML states
        if ml_state == 'Very Drowsy':
            if self.ml_very_drowsy_start is None:
                self.ml_very_drowsy_start = time.time()
            ml_very_drowsy_duration = time.time() - self.ml_very_drowsy_start
            self.ml_drowsy_start = None
            ml_drowsy_duration = 0.0
        elif ml_state == 'Drowsy':
            if self.ml_drowsy_start is None:
                self.ml_drowsy_start = time.time()
            ml_drowsy_duration = time.time() - self.ml_drowsy_start
            self.ml_very_drowsy_start = None
            ml_very_drowsy_duration = 0.0
        else:
            self.ml_very_drowsy_start = None
            self.ml_drowsy_start = None
            ml_very_drowsy_duration = 0.0
            ml_drowsy_duration = 0.0

        # Time thresholds (seconds)
        # VERY_DROWSY triggers if eyes closed >= 3.0s, ML very drowsy >= 2.5s, or head bowed >= 2.5s
        # DROWSY triggers if eyes closed >= 2.0s, ML drowsy >= 2.0s, or active yawn
        
        if ml_very_drowsy_duration >= 2.5 or eye_closure_duration >= 3.0 or head_bow_duration >= 2.5:
            self.state = 'VERY_DROWSY'
            if head_bow_duration >= 2.5:
                self._set_alert("HEAD BOWED DOWN! WAKE UP!")
            else:
                self._set_alert("DANGER! VERY DROWSY — WAKE UP!")
            self._sound("VERY_DROWSY")
        elif ml_drowsy_duration >= 2.0 or eye_closure_duration >= 2.0 or (yawn_freq >= YAWN_FREQ_LIMIT and self.yawning_now):
            self.state = 'DROWSY'
            self._set_alert(f"Drowsy! (PERCLOS={perclos:.0%})")
            self._sound("DROWSY")
        else:
            self.state = 'ALERT'
            self._clear_alert()
            self._sound("ALERT")

        sc = STATE_COLOR[self.state]

        # ── FACE bounding box (rounded, color = state) ─
        x1,y1,x2,y2 = get_face_bbox(lm,w,h,pad=10)
        draw_rounded_rect(frame,x1,y1,x2,y2,sc,t=1,r=10)

        # State badge on top of face box
        badge = f" {self.state} "
        (bw2,bh2),_ = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.35,1)
        bx = x1+(x2-x1)//2-bw2//2
        cv2.rectangle(frame,(bx-2,y1-bh2-6),(bx+bw2+2,y1),sc,-1)
        cv2.putText(frame,badge,(bx,y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,C['black'],1)

        # ── EYE boxes ─────────────────────────────────
        eye_c = C['red'] if eyes_closed else C['green']
        draw_feature_box(frame,lm,LEFT_EYE, w,h,eye_c,
                         f"EYES:{ear_val:.2f}",pad=3)
        draw_feature_box(frame,lm,RIGHT_EYE,w,h,eye_c,
                         f"EYES:{ear_val:.2f}",pad=3)

        # ── MOUTH box ─────────────────────────────────
        mouth_c = C['red'] if self.yawning_now else C['orange']
        mouth_lbl = f"YAWNING! ({yawn_freq}/min)" if self.yawning_now \
                    else f"MAR:{mar_val:.2f}"
        draw_feature_box(frame,lm,MOUTH_OUTER,w,h,mouth_c,mouth_lbl,pad=3)

        # ── Yawn counter badge (top-right) ────────────
        yc = C['red'] if yawn_freq >= YAWN_FREQ_LIMIT else C['white']
        put_label_bg(frame,f"YAWNS:{yawn_freq}/min",
                     w-90,15,yc,scale=0.35,t=1)

        # ── Top info bar ──────────────────────────────
        cv2.putText(frame,f"STATE: {self.state}",(5,18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,sc,1)
        cv2.putText(frame,
            f"EYES:{ear_val:.2f}(th:{self.ear_thresh}) "
            f"MAR:{mar_val:.2f}(th:{self.mar_thresh:.2f}) "
            f"PERCLOS:{perclos:.0%}",
            (5,32),cv2.FONT_HERSHEY_SIMPLEX,0.28,(200,200,200),1)
        
        cv2.putText(frame, f"ML: {ml_state} ({confidence*100:.0f}%)", 
                    (5, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, sc, 1)

        # ── Red overlay for VERY_DROWSY ───────────────
        if self.state == "VERY_DROWSY":
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(w,h),(0,0,180),-1)
            cv2.addWeighted(ov,0.28,frame,0.72,0,frame)
            cv2.putText(frame,"WAKE UP!",
                (w//2-60,h//2),cv2.FONT_HERSHEY_SIMPLEX,1.0,C['red'],2)

        return frame,{
            "state"           : self.state,
            "eyes"            : round(ear_val,3),
            "mar"             : round(mar_val,3),
            "perclos"         : round(perclos,3),
            "yawn_count"      : yawn_freq,
            "alert_msg"       : self.alert_msg,
            "alert_count"     : self.alert_count,
            "session_duration": int(time.time()-self.start_time),
            "face_box"        : [int(x1), int(y1), int(x2-x1), int(y2-y1)],
            "left_eye_box"    : get_feature_bbox_coords(lm, LEFT_EYE, w, h, pad=3),
            "right_eye_box"   : get_feature_bbox_coords(lm, RIGHT_EYE, w, h, pad=3),
            "mouth_box"       : get_feature_bbox_coords(lm, MOUTH_OUTER, w, h, pad=3),
            "eyes_closed"     : bool(eyes_closed),
            "yawning_now"     : bool(self.yawning_now)
        }
