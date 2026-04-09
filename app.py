import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# --- 1. SIVUN ASETUKSET ---
st.set_page_config(page_title="AI Kyykkyanalyysi", layout="wide")
st.title("🏋️ Optimoitu AI Kyykkyanalyysi")
st.write("Lataa video, niin tekoäly analysoi toistot ja tekniikan.")

# --- 2. KONFIGURAATIO ---
MODEL_PATH = 'pose_landmarker_lite.task' # Varmista että tämä on GitHubissa
TARGET_FPS = 15      # Analysoitava kuvataajuus
TARGET_WIDTH = 640   # Analysoitava resoluutio

# --- 3. APUFUNKTIOT ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 4. VIDEON LATAUS ---
uploaded_file = st.file_uploader("Valitse videotiedosto (MP4, MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Tallennetaan tiedosto väliaikaisesti
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Lasketaan hyppysuhde (esim. 60fps -> 15fps tarkoittaa joka 4. framea)
    skip_frames = max(1, int(original_fps / TARGET_FPS))
    
    # Käyttöliittymän elementit
    progress_bar = st.progress(0)
    status_text = st.empty()
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        frame_placeholder = st.empty()
    
    with col_stats:
        st.subheader("Analyysin tulokset")
        rep_list_placeholder = st.empty()

    # MediaPipe alustus
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    # Muuttujat analyysia varten
    counter = 0
    stage = "ylhaalla"
    min_knee_angle = 180
    rep_history = [] # Tallennetaan kyykkyjen syvyydet
    processed_count = 0

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # Hypätään turhien framejen yli
            target_frame_index = processed_count * skip_frames
            if target_frame_index >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            ret, frame = cap.read()
            if not ret: break
            
            # Resisointi (Nopeuttaa analyysia merkittävästi)
            h, w, _ = frame.shape
            aspect_ratio = h / w
            new_h = int(TARGET_WIDTH * aspect_ratio)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_h))
            
            # Prosessointi MediaPipelle
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((target_frame_index / original_fps) * 1000)
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    # Pisteet (Vasen puoli: 11, 23, 25, 27)
                    s = landmarks[11]; h_pt = landmarks[23]; k = landmarks[25]; a = landmarks[27]
                    shld = [s.x * TARGET_WIDTH, s.y * new_h]
                    hip = [h_pt.x * TARGET_WIDTH, h_pt.y * new_h]
                    knee = [k.x * TARGET_WIDTH, k.y * new_h]
                    ankl = [a.x * TARGET_WIDTH, a.y * new_h]
                    
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
                                rep_history.append(int(min_knee_angle))
                    
                    # Piirretään viivat
                    color = (0, 255, 0) if knee_angle < 100 else (0, 255, 255)
                    cv2.line(frame, (int(shld[0]), int(shld[1])), (int(hip[0]), int(hip[1])), color, 3)
                    cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 255), 2)
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), color, 3)
                    
                    # Lisätään teksti
                    cv2.putText(frame, f"Kulma: {int(knee_angle)}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # --- VISUAALINEN PÄIVITYS ---
            # Näytetään live-esikatselu
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Päivitetään toistolista livenä
            with rep_list_placeholder.container():
                for i, angle in enumerate(rep_history):
                    st.write(f"Toisto {i+1}: Syvyys {angle}°")

            # Päivitetään edistymispalkki
            processed_count += 1
            progress = min(1.0, target_frame_index / total_frames)
            progress_bar.progress(progress)
            status_text.text(f"Analysoitu {target_frame_index}/{total_frames} framea...")
            
            # TÄMÄ ON TÄRKEÄ: Pieni tauko antaa Streamlitille aikaa piirtää kuva
            time.sleep(0.001)

    cap.release()
    st.success(f"Analyysi valmis! Yhteensä {counter} toistoa.")
