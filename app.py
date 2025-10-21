import os
import logging
import warnings
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title("😴 Drowsiness Detection System")
st.markdown("Monitor your alertness in real-time using your webcam and AI-powered eye tracking.")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_indices]
    v1 = euclidean_distance(points[1], points[5])
    v2 = euclidean_distance(points[2], points[4])
    h1 = euclidean_distance(points[0], points[3])
    EAR = (v1 + v2) / (2.0 * h1)
    return EAR
def beep_sound():
    winsound.Beep(1000, 500)  
st.sidebar.header("⚙️ Controls")
start_button = st.sidebar.button("▶️ Start Detection")
stop_button = st.sidebar.button("⏹️ Stop Detection")
show_ear = st.sidebar.checkbox("👁️ Show EAR Value", value=True)
sound_alert = st.sidebar.checkbox("🔊 Enable Sound Alert", value=True)
ear_threshold = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.25, 0.01)
closed_time_limit = st.sidebar.slider("Closed Eyes Duration (sec)", 1, 5, 3)

FRAME_WINDOW = st.empty()
if start_button:
    cap = cv2.VideoCapture(0)
    closed_eyes_start = None
    alarm_on = False

    st.warning("**Press ‘Stop Detection’ to exit.**")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Could not access webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        h, w, _ = frame.shape

        EAR = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            left_EAR = eye_aspect_ratio(face_landmarks, LEFT_EYE, w, h)
            right_EAR = eye_aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
            EAR = (left_EAR + right_EAR) / 2.0
            for idx in LEFT_EYE + RIGHT_EYE:
                x, y = int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            if show_ear and EAR is not None:
                cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if EAR < ear_threshold:
                if closed_eyes_start is None:
                    closed_eyes_start = time.time()
                else:
                    elapsed = time.time() - closed_eyes_start
                    if elapsed >= closed_time_limit:
                        cv2.putText(frame, "⚠️ WAKE UP!", (100, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        if sound_alert:
                            beep_sound()
            else:
                closed_eyes_start = None
                alarm_on = False

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Exit condition
        if stop_button:
            break

    cap.release()
    st.success("✅ Detection stopped.")

else:
    st.info("👆 Click ‘Start Detection’ from the sidebar to begin.")
