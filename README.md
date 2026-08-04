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
├── requirements.txt         ← Python dependencies
├── Dataset/
│   └── dataset_link.txt     ← YawDD + NTHU-DDD download links
├── Output_Images/           ← Run notebook to generate these
│   ├── 1_Class_Distribution_EDA.png
│   ├── 2_Feature_Distributions_EDA.png
│   ├── 3_Correlation_Heatmap_Visualization.png
│   ├── 4_EAR_Over_Time_Visualization.png
│   ├── 5_Accuracy_Metrics_Graph.png
│   ├── 6_Confusion_Matrix.png
│   ├── 7_ROC_Curves.png
│   └── 8_Prediction_Results.png
├── detector.py          ← Core YawDD-aligned detector
├── feature_extractor.py ← EAR, MAR, Head Pose functions
├── alert_engine.py      ← Audio alert engine
├── main.py              ← FastAPI backend server
├── config.yaml          ← All thresholds & settings
├── static/              ← CSS & JS files
├── templates/           ← HTML frontend
├── best_model.pkl       ← Pre-trained SVM model
└── Demo_Video.mp4       ← Screen recording (5-10 min)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install mediapipe==0.10.9
pip install -r requirements.txt
```

### 2. Run Web Interface (FastAPI Server)
```bash
python main.py
```
*Then open your browser to `http://127.0.0.1:8000`*

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
| Classification | SVM + YawDD rule-based engine |
| Calibration | Personalized EAR/MAR per driver |
| Alert System | pygame (beep + alarm) + OpenCV overlay |
| Dashboard | FastAPI, WebSockets, HTML/JS/CSS |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94.2% |
| Precision | 93.9% |
| Recall | 94.2% |
| F1-Score | 94.0% |
| AUC-ROC | 0.981 |

---

## 📧 Contact

**Muhammad Azeen Waqas**
Institution: COMSATS University Islamabad, Wah Campus
Program: BS Artificial Intelligence (Batch 2023–27)

---

*Submitted for CoderAxo AI/ML Internship Program 2026*
*Detection method based on: Abtahi et al. (2014) YawDD: A Yawning Detection Dataset*
"# drowsiness-detection" 
