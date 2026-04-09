import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Kyykkyanalyysi", layout="wide")
st.title("🏋️ Kyykkyanalyysi")

MODEL_PATH = 'pose_landmarker_lite.task'
TARGET_FPS = 10 # Vieläkin hieman hitaampi, jotta on varmempaa

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

uploaded_file = st.file_uploader("Lataa video", type=['mp4', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = max(1, int(fps / TARGET_FPS))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Analyysimuuttujat
    counter = 0
    stage = "ylhaalla"
    rep_images = [] # Tänne tallennetaan kuvat onnistuneista kyykyistä

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        curr_frame = 0
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame * skip_frames)
            ret, frame = cap.read()
            if not ret: break
            
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / fps) * 1000)
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    # Lasketaan lonkka, polvi, nilkka (vasen puoli)
                    h_pt = landmarks[23]; k_pt = landmarks[25]; a_pt = landmarks[27]
                    hip = [h_pt.x * w, h_pt.y * h]
                    knee = [k_pt.x * w, k_pt.y * h]
                    ankl = [a_pt.x * w, a_pt.y * h]
                    
                    angle = calculate_angle(hip, knee, ankl)
                    
                    if angle < 140 and stage == "ylhaalla":
                        stage = "alhaalla"
                        min_angle = 180
                    
                    if stage == "alhaalla":
                        min_angle = min(min_angle, angle)
                        if angle > 155:
                            stage = "ylhaalla"
                            counter += 1
                            # Tallennetaan kuva tästä hetkestä (vähän ennen nousua)
                            # Piirretään kulma kuvaan
                            cv2.putText(frame, f"Syvyys: {int(min_angle)}deg", (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            rep_images.append((counter, int(min_angle), frame.copy()))

            curr_frame += 1
            progress_bar.progress(min(1.0, (curr_frame * skip_frames) / total_frames))
            status_text.text(f"Analysoitu {counter} toistoa...")

    cap.release()
    st.success(f"Valmis! Löytyi {counter} toistoa.")

    # Näytetään tulokset kuvina
    if rep_images:
        st.subheader("Toistojen kohokohdat")
        for num, ang, img in rep_images:
            st.write(f"Toisto {num} - Syvin kohta: {ang}°")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
