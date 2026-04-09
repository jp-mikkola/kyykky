import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# --- ASETUKSET ---
st.set_page_config(page_title="Turbo Kyykkyanalyysi", layout="wide")
st.title("🚀 Turbo-optimoitu Kyykkyanalyysi")

# KEVENNYS 1: Käytetään Lite-mallia
MODEL_PATH = 'pose_landmarker_lite.task'
# KEVENNYS 2: Analysoidaan vain 5 kuvaa sekunnissa
ANALYSIS_FPS = 5 

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
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # KEVENNYS 3: Lasketaan hyppysuhde
    skip_frames = max(1, int(orig_fps / ANALYSIS_FPS))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # MediaPipe asetukset (Äärimmäinen kevennys)
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.2, # Höllennetään vaatimuksia nopeuden takia
        min_pose_presence_confidence=0.2,
        min_tracking_confidence=0.2
    )

    counter = 0
    stage = "ylhaalla"
    rep_history = []

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            # Hypätään suuren osan yli
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * skip_frames)
            ret, frame = cap.read()
            if not ret: break
            
            # KEVENNYS 4: Pienennetään resoluutio erittäin pieneksi (esim 320px)
            # MediaPipe Pose toimii loistavasti jopa 256x256 kuvalla
            frame = cv2.resize(frame, (320, int(320 * (frame.shape[0]/frame.shape[1]))))
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / orig_fps) * 1000)
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    # Lonkka (23), Polvi (25), Nilkka (27)
                    h, w = frame.shape[:2]
                    hip = [landmarks[23].x * w, landmarks[23].y * h]
                    knee = [landmarks[25].x * w, landmarks[25].y * h]
                    ankl = [landmarks[27].x * w, landmarks[27].y * h]
                    
                    angle = calculate_angle(hip, knee, ankl)
                    
                    if angle < 140 and stage == "ylhaalla":
                        stage = "alhaalla"
                        min_angle = 180
                    if stage == "alhaalla":
                        min_angle = min(min_angle, angle)
                        if angle > 155:
                            stage = "ylhaalla"
                            counter += 1
                            rep_history.append(int(min_angle))

            frame_idx += 1
            progress_bar.progress(min(1.0, (frame_idx * skip_frames) / total_frames))
            status_text.text(f"Prosessoitu {int((frame_idx * skip_frames) / total_frames * 100)}%...")

    cap.release()
    st.success("Analyysi valmis!")
    
    # Näytetään tulokset
    col1, col2 = st.columns(2)
    col1.metric("Toistot yhteensä", counter)
    
    if rep_history:
        col2.write("Toistojen syvyydet (astetta):")
        for i, a in enumerate(rep_history):
            st.write(f"Toisto {i+1}: **{a}°**")
