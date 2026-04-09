import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# --- 1. SIVUN ASETUKSET ---
st.set_page_config(page_title="AI Kyykkyanalyysi", layout="wide")
st.title("🏋️ AI Kyykkyanalyysi - Lataa video")
st.write("Kuvaa kyykkysi suoraan sivulta ja lataa tiedosto alle.")

# --- 2. MEDIAPIPE SETUP (Nyt voimme käyttää Heavy-mallia!) ---
MODEL_PATH = 'pose_landmarker_heavy.task'

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 3. VIDEON LATAUS ---
uploaded_file = st.file_uploader("Valitse videotiedosto...", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Tallennetaan ladattu tiedosto väliaikaisesti, jotta OpenCV voi lukea sen
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info("Analysoidaan videota... Odota hetki.")
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    
    # Alustetaan MediaPipe
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
    
    # Analyysimuuttujat
    counter = 0
    stage = "ylhaalla"
    min_knee_angle = 180
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        curr_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Lasketaan aikaleima FPS:n perusteateella
            timestamp_ms = int((curr_frame / fps) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    # Poimitaan pisteet (oletus: vasen puoli)
                    s = landmarks[11]; h_pt = landmarks[23]; k = landmarks[25]; a = landmarks[27]
                    shld = [s.x * w, s.y * h]
                    hip = [h_pt.x * w, h_pt.y * h]
                    knee = [k.x * w, k.y * h]
                    ankl = [a.x * w, a.y * h]
                    
                    knee_angle = calculate_angle(hip, knee, ankl)
                    
                    # Logiikka
                    if knee_angle < 140 and stage == "ylhaalla":
                        stage = "alhaalla"
                        min_knee_angle = 180
                    if stage == "alhaalla":
                        min_knee_angle = min(min_knee_angle, knee_angle)
                        if knee_angle > 155:
                            stage = "ylhaalla"
                            if min_knee_angle < 140:
                                counter += 1

                    # Piirretään grafiikat
                    cv2.line(frame, (int(shld[0]), int(shld[1])), (int(hip[0]), int(hip[1])), (0, 255, 0), 5)
                    cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 255), 3)
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), (0, 255, 0), 5)
                    
                    cv2.putText(frame, f"REPS: {counter}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                    cv2.putText(frame, f"KNEE: {int(knee_angle)}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Näytetään prosessoitu frame
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            curr_frame += 1
            progress_bar.progress(curr_frame / total_frames)

    st.success(f"Analyysi valmis! Yhteensä {counter} toistoa.")
    cap.release()
