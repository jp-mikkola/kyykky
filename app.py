import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import math

# --- 1. SIVUN ASETUKSET ---
st.set_page_config(page_title="AI Kyykkyvalmentaja", layout="wide")
st.title("🏋️ AI Kyykkyvalmentaja (Moderni Tasks-versio)")

# --- 2. APUFUNKTIOT ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def calculate_vertical_angle(top, bottom):
    radians = np.arctan2(top[0] - bottom[0], bottom[1] - top[1])
    return radians * 180.0 / np.pi

# --- 3. MEDIAPIPE TASKS ALUSTUS ---
# Huom: Streamlitissä model_path on oltava samassa kansiossa kuin app.py
model_path = 'pose_landmarker_heavy.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Alustetaan session_state, jotta toistot pysyvät muistissa selaimen virkistyessä
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'rep_history' not in st.session_state:
    st.session_state.rep_history = []

# --- 4. SIVUPALKKI OHJAUKSEEN ---
run = st.sidebar.toggle('Käynnistä Kamera / Analyysi', value=False)
if st.sidebar.button('Nollaa toistot'):
    st.session_state.counter = 0
    st.session_state.rep_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.info("Tämä versio käyttää modernia MediaPipe Tasks -moottoria.")

# --- 5. NÄKYMÄN ALUSTUS ---
col1, col2 = st.columns([2, 1])
with col1:
    FRAME_WINDOW = st.image([])
with col2:
    st.subheader("Toistot: " + str(st.session_state.counter))
    stats_placeholder = st.empty()

# --- 6. MUUTTUJAT ---
stage = "ylhaalla"
min_knee_angle = 180
is_ascending = False
shoulder_points_lasku = []
shoulder_points_nousu = []
metrics = {
    "lasku": {"drift": 0, "lean_diff": 0, "neck_diff": 0},
    "nousu": {"drift": 0, "lean_diff": 0, "neck_diff": 0}
}

# --- 7. PÄÄSILMUKKA ---
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while run:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Käytetään nykyistä aikaa leimana
        frame_timestamp_ms = int(time.time() * 1000)
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if pose_landmarker_result.pose_landmarks:
            for landmarks in pose_landmarker_result.pose_landmarks:
                
                # Puolen tunnistus
                left_vis = sum([landmarks[i].visibility for i in [11, 23, 25, 27, 31]])
                right_vis = sum([landmarks[i].visibility for i in [12, 24, 26, 28, 32]])
                facing_left = left_vis > right_vis
                
                if facing_left:
                    s, h_pt, k, a, t, e = landmarks[11], landmarks[23], landmarks[25], landmarks[27], landmarks[31], landmarks[7]
                    direction_mult = 1
                else:
                    s, h_pt, k, a, t, e = landmarks[12], landmarks[24], landmarks[26], landmarks[28], landmarks[32], landmarks[8]
                    direction_mult = -1

                if all(p.visibility > 0.4 for p in [s, h_pt, k, a, t]):
                    shld = [s.x * w, s.y * h]
                    hip = [h_pt.x * w, h_pt.y * h]
                    knee = [k.x * w, k.y * h]
                    ankl = [a.x * w, a.y * h]
                    mid_x = (ankl[0] + (t.x * w)) / 2
                    ear = [e.x * w, e.y * h] if e.visibility > 0.3 else None

                    # Laskenta
                    knee_angle = calculate_angle(hip, knee, ankl)
                    back_angle = calculate_vertical_angle(shld, hip)
                    shin_angle = calculate_vertical_angle(knee, ankl)
                    
                    current_lean_diff = (back_angle - shin_angle) * direction_mult
                    current_drift = (shld[0] - mid_x) * direction_mult
                    current_neck_diff = (calculate_vertical_angle(ear, shld) - back_angle) * direction_mult if ear else 0

                    # Seuranta
                    if stage == "alhaalla":
                        p = "nousu" if is_ascending else "lasku"
                        if abs(current_drift) > abs(metrics[p]["drift"]): metrics[p]["drift"] = current_drift
                        if abs(current_lean_diff) > abs(metrics[p]["lean_diff"]): metrics[p]["lean_diff"] = current_lean_diff
                        if abs(current_neck_diff) > abs(metrics[p]["neck_diff"]): metrics[p]["neck_diff"] = current_neck_diff

                        if not is_ascending:
                            shoulder_points_lasku.append((int(shld[0]), int(shld[1])))
                            if knee_angle > min_knee_angle + 2: is_ascending = True
                        else:
                            shoulder_points_nousu.append((int(shld[0]), int(shld[1])))
                        min_knee_angle = min(min_knee_angle, knee_angle)

                    if knee_angle < 140 and stage == "ylhaalla":
                        stage = "alhaalla"; is_ascending = False; min_knee_angle = 180
                        metrics = {"lasku": {"drift": 0, "lean_diff": 0, "neck_diff": 0},
                                   "nousu": {"drift": 0, "lean_diff": 0, "neck_diff": 0}}
                    
                    elif knee_angle > 155 and stage == "alhaalla":
                        stage = "ylhaalla"
                        if min_knee_angle < 140:
                            st.session_state.counter += 1
                            rep_info = f"Toisto {st.session_state.counter}: Syvyys {int(min_knee_angle)}°"
                            st.session_state.rep_history.append(rep_info)

                    # --- LIVE VÄRITYS JA PIIRTO ---
                    body_color = (0, 0, 255) if abs(current_lean_diff) > 15 else (255, 255, 255)
                    neck_color = (0, 0, 255) if abs(current_neck_diff) > 15 else (255, 200, 0)

                    cv2.line(frame, (int(mid_x), 0), (int(mid_x), h), (100, 100, 100), 1)
                    cv2.line(frame, (int(shld[0]), int(shld[1])), (int(hip[0]), int(hip[1])), body_color, 3)
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), body_color, 3)
                    if ear: cv2.line(frame, (int(ear[0]), int(ear[1])), (int(shld[0]), int(shld[1])), neck_color, 3)

        # Päivitetään selainnäkymä
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Päivitetään historia sarakkeeseen
        with stats_placeholder.container():
            for item in reversed(st.session_state.rep_history[-5:]):
                st.write(item)

cap.release()