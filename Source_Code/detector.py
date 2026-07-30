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
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'best_model.pkl')
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
# Based on YawDD + Soukupova 2016 research values
EAR_OPEN        = 0.30   # eyes fully open (calibrated from user)
EAR_CLOSED      = 0.22   # eyes closed threshold
MAR_YAWN        = 0.65   # yawning threshold (YawDD benchmark)
PERCLOS_LIMIT   = 0.20   # 20% eye closure in 60 frames = drowsy
YAWN_FREQ_LIMIT = 3      # 3 yawns in 60 sec = drowsy (YawDD)
YAWN_FRAMES     = 15     # frames mouth must be open to count as yawn
DROWSY_FRAMES   = 48     # 2 seconds of closed eyes before DROWSY
HARD_FRAMES     = 96     # 4 seconds before VERY_DROWSY
NO_FACE_LIMIT   = 80     # frames no face before alert (~4 sec)

# ─── Sound Engine ─────────────────────────────────────
def _gen_wav(path, freq=1000, dur=0.4, vol=0.7, rate=44100):
    with wave.open(path, 'w') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        for i in range(int(rate * dur)):
            v = int(vol * 32767 * math.sin(2 * math.pi * freq * i / rate))
            f.writeframes(struct.pack('<h', v))

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
import mediapipe.python.solutions.face_mesh as mp_face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True,
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
        self.ear_thresh      = EAR_CLOSED
        self.mar_thresh      = MAR_YAWN

        # YawDD-style counters
        self.ear_history     = []   # rolling 60-frame EAR window
        self.yawn_counter    = 0    # current yawn frame count
        self.yawn_events     = []   # timestamps of completed yawns
        self.yawning_now     = False
        self.pitch_smooth    = 0.0  # EMA smoothed head pitch

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
        if state == self._last_snd: return
        self._last_snd = state
        sound.stop()
        if   state == "DROWSY":      sound.play(BEEP_PATH,  loop=True)
        elif state == "VERY_DROWSY": sound.play(ALARM_PATH, loop=True)
        elif state == "AWAY":        sound.play(AWAY_PATH,  loop=True)

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
            pct = int(n/60*100)
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
                self.calib_ear.append(e)
                self.calib_mar.append(m)

            if n >= 60:
                base_ear        = np.mean(self.calib_ear)
                base_mar        = np.mean(self.calib_mar)
                # Cap the threshold so unusually high baseline EARs don't cause PERCLOS to break
                self.ear_thresh = min(0.23, max(0.16, round(base_ear - 0.080, 3)))
                self.mar_thresh = max(0.55, round(base_mar + 0.22,  3))
                self.calibrated = True
                print(f"Calibrated -> EAR thresh:{self.ear_thresh}  "
                      f"MAR thresh:{self.mar_thresh}  "
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

            if self.no_face_counter >= 90:
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
            if self.no_face_counter >= 90:
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
            self.yawn_counter = max(0, self.yawn_counter-2)
            if self.yawn_counter == 0:
                self.yawning_now = False

        yawn_freq = self._yawn_freq()

        # ── Eye closure counter ───────────────────────
        eyes_closed = ear_val < self.ear_thresh
        if eyes_closed: self.frame_counter += 1
        else:           self.frame_counter = max(0, self.frame_counter-16)

        # ── YawDD drowsiness classification ──────────
        # ML Model Inference
        raw_pitch = get_head_pitch(lm, w, h)
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
            pred_label = ml_model.predict(features)[0]
            pred_proba = ml_model.predict_proba(features)[0]
            confidence = pred_proba[pred_label]
            ml_state = CLASSES[pred_label]
        else:
            ml_state = 'Alert'
            confidence = 1.0

        if not hasattr(self, 'ml_very_drowsy_frames'):
            self.ml_very_drowsy_frames = 0
            self.ml_drowsy_frames = 0

        if ml_state == 'Very Drowsy':
            self.ml_very_drowsy_frames += 1
            self.ml_drowsy_frames += 1
        elif ml_state == 'Drowsy':
            self.ml_drowsy_frames += 1
            self.ml_very_drowsy_frames = max(0, self.ml_very_drowsy_frames - 5)
        else:
            self.ml_very_drowsy_frames = max(0, self.ml_very_drowsy_frames - 10)
            self.ml_drowsy_frames = max(0, self.ml_drowsy_frames - 10)

        if self.ml_very_drowsy_frames >= 30 or self.frame_counter >= HARD_FRAMES:
            self.state = 'VERY_DROWSY'
            self._set_alert("DANGER! VERY DROWSY — WAKE UP!")
            self._sound("VERY_DROWSY")
        elif self.ml_drowsy_frames >= 20 or self.frame_counter >= DROWSY_FRAMES or yawn_freq >= YAWN_FREQ_LIMIT:
            self.state = 'DROWSY'
            self._set_alert(f"Drowsy! (PERCLOS={perclos:.0%})")
            self._sound("DROWSY")
        else:
            prev = self.state
            self.state = 'ALERT'
            self._clear_alert()
            if prev != "ALERT": self._sound("ALERT")

        sc = STATE_COLOR[self.state]

        # ── FACE bounding box (rounded, color = state) ─
        x1,y1,x2,y2 = get_face_bbox(lm,w,h,pad=18)
        draw_rounded_rect(frame,x1,y1,x2,y2,sc,t=2,r=18)

        # State badge on top of face box
        badge = f" {self.state} "
        (bw2,bh2),_ = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.58,2)
        bx = x1+(x2-x1)//2-bw2//2
        cv2.rectangle(frame,(bx-4,y1-bh2-10),(bx+bw2+4,y1),sc,-1)
        cv2.putText(frame,badge,(bx,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.58,C['black'],2)

        # ── EYE boxes ─────────────────────────────────
        eye_c = C['red'] if eyes_closed else C['green']
        draw_feature_box(frame,lm,LEFT_EYE, w,h,eye_c,
                         f"EYES:{ear_val:.2f}",pad=6)
        draw_feature_box(frame,lm,RIGHT_EYE,w,h,eye_c,
                         f"EYES:{ear_val:.2f}",pad=6)

        # ── MOUTH box ─────────────────────────────────
        mouth_c = C['red'] if self.yawning_now else C['orange']
        mouth_lbl = f"YAWNING! ({yawn_freq}/min)" if self.yawning_now \
                    else f"MAR:{mar_val:.2f}"
        draw_feature_box(frame,lm,MOUTH_OUTER,w,h,mouth_c,mouth_lbl,pad=5)

        # ── Yawn counter badge (top-right) ────────────
        yc = C['red'] if yawn_freq >= YAWN_FREQ_LIMIT else C['white']
        put_label_bg(frame,f"YAWNS:{yawn_freq}/min",
                     w-160,30,yc,scale=0.52,t=1)

        # ── Top info bar ──────────────────────────────
        cv2.putText(frame,f"STATE: {self.state}",(10,36),
                    cv2.FONT_HERSHEY_SIMPLEX,0.88,sc,2)
        cv2.putText(frame,
            f"EYES:{ear_val:.3f}(th:{self.ear_thresh})  "
            f"MAR:{mar_val:.3f}(th:{self.mar_thresh:.2f})  "
            f"PERCLOS:{perclos:.0%}  "
            f"FC:{self.frame_counter}/{DROWSY_FRAMES}",
            (10,66),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)
        
        cv2.putText(frame, f"ML: {ml_state} ({confidence*100:.0f}%)", 
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sc, 1)

        # ── Red overlay for VERY_DROWSY ───────────────
        if self.state == "VERY_DROWSY":
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(w,h),(0,0,180),-1)
            cv2.addWeighted(ov,0.28,frame,0.72,0,frame)
            cv2.putText(frame,"WAKE UP!",
                (w//2-140,h//2),cv2.FONT_HERSHEY_SIMPLEX,2.2,C['red'],5)

        return frame,{
            "state"           : self.state,
            "eyes"            : round(ear_val,3),
            "mar"             : round(mar_val,3),
            "perclos"         : round(perclos,3),
            "yawn_count"      : yawn_freq,
            "alert_msg"       : self.alert_msg,
            "alert_count"     : self.alert_count,
            "session_duration": int(time.time()-self.start_time)
        }
