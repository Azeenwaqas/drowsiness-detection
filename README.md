# 🚗 Real-Time Driver Drowsiness Detection
### CoderAxo AI/ML Internship — Offer ID: CAX-OL-2026-290

**Author:** Muhammad Azeen Waqas
**Institution:** COMSATS University Islamabad, Wah Campus
**Program:** BS Artificial Intelligence (Batch 2023–27)
**Submission Deadline:** 01-08-2026

---

## 📌 Project Overview

DrowsyGuard is a real-time computer vision system that detects driver drowsiness using facial landmark analysis based on the **YawDD Dataset** methodology. It monitors:

- 👁 **Eye Aspect Ratio (EAR)** — eye closure detection
- 👄 **Mouth Aspect Ratio (MAR)** — yawn detection
- 📊 **PERCLOS** — % eye closure over 60-frame rolling window
- 😮 **Yawn Frequency** — yawn count per 60 seconds (YawDD metric)
- 🔄 **Head Pose** — head down / face turned detection

---

## 📁 Folder Structure

```
DrowsyGuard/
├── Notebook.ipynb           ← Primary submission (Jupyter Notebook)
├── Report.pdf               ← Project Report (12 sections, 12 pages)
├── README.md                ← This file
├── Dataset/                 ← Contains dataset details and link
│   └── dataset_link.txt
├── Output_Images/           ← Visualizations and performance graphs
├── Source_Code/             ← Complete application codebase
│   ├── backend/             ← FastAPI backend with model integration
│   │   ├── main.py          ← FastAPI WebSocket server
│   │   ├── detector.py      ← Core driver monitoring logic
│   │   ├── best_model.pkl   ← Pre-trained SVM model
│   │   └── requirements.txt ← Backend dependencies
│   └── frontend/            ← Front-end interface files
│       ├── index.html
│       └── static/
│           ├── script.js    ← Local video canvas & client drawing
│           └── style.css    ← UI dashboard stylesheet
└── Demo_Video.mp4           ← Screen recording (5-10 min)
```

---

## 🚀 Quick Start

### 🌐 Live Demo
Try the live production application here: **[https://drowsiness-detection-omega.vercel.app/](https://drowsiness-detection-omega.vercel.app/)**

### 1. Install dependencies
```bash
pip install mediapipe==0.10.11
pip install -r requirements.txt
```

### 2. Run Web Interface (FastAPI Server)
```bash
python backend/main.py
```
*Then open your browser to the local URL provided by Uvicorn.*

### 3. Run Jupyter Notebook (ML Pipeline)
```bash
jupyter notebook Notebook.ipynb
```

---

## 🧠 Detection Logic (YawDD Dataset Method)

| Signal | Method | Threshold | Source |
|--------|--------|-----------|--------|
| Eye closure | PERCLOS over 60 frames | > 15% | Wierwille 1994 |
| Yawn detection | MAR sustained > 15 frames | MAR > 0.65 | YawDD |
| Yawn frequency | Count per 60 seconds | ≥ 3/min | YawDD benchmark |
| Head absence | Face not visible | > 4 seconds | Custom |

### States
```
🟢 ALERT       → Normal driving
🟡 DROWSY      → Eyes closing OR yawning frequently
🔴 VERY DROWSY → Eyes closed 2+ seconds → WAKE UP!
```

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Face Detection | MediaPipe FaceMesh (468 landmarks) |
| Feature Extraction | EAR, MAR (dual), PERCLOS, Head Pose |
| Classification | SVM (Tuned via GridSearchCV) + YawDD rules |
| Calibration | Personalized EAR/MAR per driver |
| Alert System | Web Audio API (Client-side Synthesizer) |
| Dashboard | FastAPI, WebSockets, HTML5 Canvas, JS/CSS |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.4% |
| Precision | 98.4% |
| Recall | 98.4% |
| F1-Score | 98.4% |
| AUC-ROC | 0.998 |

---

## 📧 Contact

**Muhammad Azeen Waqas**
Institution: COMSATS University Islamabad, Wah Campus
Program: BS Artificial Intelligence (Batch 2023–27)

---

*Submitted for CoderAxo AI/ML Internship Program 2026*
*Detection method based on: Abtahi et al. (2014) YawDD: A Yawning Detection Dataset*
"# drowsiness-detection" 
