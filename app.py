import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

st.set_page_config(page_title="AI Kyykkyvalmentaja", layout="wide")
st.title("🏋️ AI Kyykkyanalyysi - Video-output")

MODEL_PATH = 'pose_landmarker_lite.task'
TARGET_FPS = 15 
TARGET_WIDTH = 640

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
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, int(orig_fps / TARGET_FPS))
    
    # Valmistellaan output-video
    # Käytetään 'avc1' (H.264) tai 'mp4v'. Jos kumpikaan ei toimi pilvessä, 
    # se on merkki siitä että ffmpeg-kirjastot puuttuvat.
    output_path = "analysoitu_kyykky.mp4"
# 'avc1' on H.264-koodekin tunnus, jota selaimet rakastavat
fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = None

    progress_bar = st.progress(0)
    status = st.empty()

    # MediaPipe
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    rep_count = 0
    stage = "ylhaalla"

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * skip_frames)
            ret, frame = cap.read()
            if not ret: break
            
            # Resisointi
            h, w = frame.shape[:2]
            new_h = int(TARGET_WIDTH * (h/w))
            # H.264 vaatii parilliset mitat
            new_h = new_h if new_h % 2 == 0 else new_h - 1
            frame = cv2.resize(frame, (TARGET_WIDTH, new_h))
            
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (TARGET_WIDTH, new_h))

            # Analyysi
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / orig_fps) * 1000)
            results = landmarker.detect_for_video(mp_image, ts)
            
            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    # Pisteet
                    hip = [landmarks[23].x * TARGET_WIDTH, landmarks[23].y * new_h]
                    knee = [landmarks[25].x * TARGET_WIDTH, landmarks[25].y * new_h]
                    ankl = [landmarks[27].x * TARGET_WIDTH, landmarks[27].y * new_h]
                    shld = [landmarks[11].x * TARGET_WIDTH, landmarks[11].y * new_h]
                    
                    angle = calculate_angle(hip, knee, ankl)
                    
                    # Logiikka
                    if angle < 140 and stage == "ylhaalla": stage = "alhaalla"
                    if angle > 155 and stage == "alhaalla":
                        stage = "ylhaalla"
                        rep_count += 1
                    
                    # Piirto
                    color = (0, 255, 0) if angle < 110 else (0, 255, 255)
                    cv2.line(frame, (int(shld[0]), int(shld[1])), (int(hip[0]), int(hip[1])), color, 3)
                    cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 255), 2)
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), color, 3)
                    cv2.putText(frame, f"Kulma: {int(angle)}", (20, 40), 1, 1.5, (255, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
            progress_bar.progress(min(1.0, (frame_idx * skip_frames) / total_frames))
            status.text(f"Analysoidaan... Toistot: {rep_count}")

    cap.release()
    if out: out.release()

    # TÄRKEÄÄ: Streamlit Cloud/Selain ei usein näytä 'mp4v' suoraan.
    # Jos video ei näy, se täytyy ladata koneelle katsottavaksi.
    st.success(f"Analyysi valmis! Toistoja: {rep_count}")
    
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            st.video(f.read())
            st.download_button("Lataa analysoitu video", f, "analysoitu.mp4")
