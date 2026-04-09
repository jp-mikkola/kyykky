import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# --- ASETUKSET ---
st.set_page_config(page_title="Optimoitu Kyykkyanalyysi", layout="wide")
st.title("⚡ Nopeampi AI Kyykkyanalyysi")

# Käytetään Lite-mallia nopeuden varmistamiseksi pilvessä
MODEL_PATH = 'pose_landmarker_lite.task'
TARGET_FPS = 15  # Tavoite-FPS analyysille
TARGET_WIDTH = 640 # Pienennetään resoluutiota prosessointia varten

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

uploaded_file = st.file_uploader("Lataa kyykkyvideo", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Lasketaan kuinka monta framea hypätään yli
    # Esim. 60 FPS / 15 FPS = 4 (analysoidaan joka 4. frame)
    skip_frames = max(1, int(original_fps / TARGET_FPS))
    
    st.info(f"Videon alkuperäinen FPS: {int(original_fps)}. Analysoidaan {TARGET_FPS} kuvaa sekunnissa.")
    
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    
    # MediaPipe alustus
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    counter = 0
    stage = "ylhaalla"
    min_knee_angle = 180
    processed_count = 0

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # Asetetaan videon lukupää oikeaan kohtaan (hyppy)
            cap.set(cv2.CAP_PROP_POS_FRAMES, processed_count * skip_frames)
            ret, frame = cap.read()
            if not ret: break
            
            # 1. PIENENNETÄÄN KUVA
            h, w, _ = frame.shape
            aspect_ratio = h / w
            new_h = int(TARGET_WIDTH * aspect_ratio)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_h))
            
            # MediaPipe vaatii RGB:n
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Aikaleima ms
            timestamp_ms = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / original_fps) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    s = landmarks[11]; h_pt = landmarks[23]; k = landmarks[25]; a = landmarks[27]
                    
                    # Pisteet uudessa resoluutiossa
                    hip = [h_pt.x * TARGET_WIDTH, h_pt.y * new_h]
                    knee = [k.x * TARGET_WIDTH, k.y * new_h]
                    ankl = [a.x * TARGET_WIDTH, a.y * new_h]
                    shld = [s.x * TARGET_WIDTH, s.y * new_h]

                    knee_angle = calculate_angle(hip, knee, ankl)
                    
                    # Kyykkylogiikka
                    if knee_angle < 140 and stage == "ylhaalla":
                        stage = "alhaalla"
                        min_knee_angle = 180
                    if stage == "alhaalla":
                        min_knee_angle = min(min_knee_angle, knee_angle)
                        if knee_angle > 155:
                            stage = "ylhaalla"
                            if min_knee_angle < 140:
                                counter += 1

                    # Piirretään grafiikat pienempään kuvaan
                    cv2.line(frame, (int(shld[0]), int(shld[1])), (int(hip[0]), int(hip[1])), (0, 255, 0), 3)
                    cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 255), 2)
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), (0, 255, 0), 3)
                    cv2.putText(frame, f"REPS: {counter}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Päivitetään kuva selaimeen
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            processed_count += 1
            # Päivitetään edistyminen (arvioitu)
            current_progress = min(1.0, (processed_count * skip_frames) / total_frames)
            progress_bar.progress(current_progress)

    st.success(f"Analyysi valmis! Toistot: {counter}")
    cap.release()
