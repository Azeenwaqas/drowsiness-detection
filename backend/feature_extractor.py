"""
Feature Extraction Module — DrowsyGuard
CoderAxo Internship CAX-OL-2026-290
Author: Muhammad Azeen Waqas
Institution: COMSATS University Islamabad, Wah Campus
"""
import cv2
import math
import numpy as np
from scipy.spatial import distance as dist

# MediaPipe landmark indices
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181]

MODEL_3D = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
], dtype=np.float64)

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices])
    A = dist.euclidean(pts[2], pts[10])
    B = dist.euclidean(pts[4], pts[8])
    C = dist.euclidean(pts[0], pts[6])
    return (A + B) / (2.0 * C + 1e-6)

def get_head_pitch(landmarks, frame_shape):
    h, w = frame_shape[:2]
    key_idxs = [1, 152, 33, 263, 61, 291]
    image_2d = np.array([(landmarks[i].x * w, landmarks[i].y * h)
                          for i in key_idxs], dtype=np.float64)
    focal = w
    cam_mat = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
    try:
        _, rvec, _ = cv2.solvePnP(MODEL_3D, image_2d, cam_mat,
                                    np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rvec)
        return math.degrees(math.asin(-rmat[2][0]))
    except:
        return 0.0
