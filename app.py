import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import av
import os

# --- ASETUKSET ---
st.set_page_config(page_title="AI Kyykkyanalyysi", layout="wide")
st.title("🏋️ AI Kyykkyanalyysi - Videoanalyysi")

MODEL_PATH = 'pose_landmarker_lite.task'
TARGET_FPS = 15
TARGET_WIDTH = 640

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

uploaded_file = st.file_uploader("Lataa kyykkyvideo analysoitavaksi", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # 1. VALMISTELUT
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    # Valmistellaan polku tulosvideolle
    output_path = "output_analyzed.mp4"
    
    cap = cv2.VideoCapture(tfile.name)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, int(original_fps / TARGET_FPS))
    
    st.info("Analysoidaan videota... Tämä on nopeampaa ilman esikatselua.")
    progress_bar = st.progress(0)
    
    # Alustetaan PyAV videon tallennusta varten (H.264 pakkaus selaimelle)
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=TARGET_FPS)
    stream.pix_fmt = "yuv420p" # Standardi muoto selaimille
    
    # MediaPipe alustus
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    counter = 0
    stage = "ylhaalla"
    rep_history = []
    processed_count = 0

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            target_frame_index = processed_count * skip_frames
            if target_frame_index >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            ret, frame = cap.read()
            if not ret: break
            
            # Resisointi
            h, w, _ = frame.shape
            aspect_ratio = h / w
            new_h = int(TARGET_WIDTH * aspect_ratio)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_h))
            
            # Alustetaan stream-asetukset ensimmäisellä framella
            if processed_count == 0:
                stream.width = TARGET_WIDTH
                stream.height = new_h

            # Prosessointi
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((target_frame_index / original_fps) * 1000)
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    s = landmarks[11]; h_pt = landmarks[23]; k = landmarks[25]; a = landmarks[27]
                    hip = [h_pt.x * TARGET_WIDTH, h_pt.y * new_h]
                    knee = [k.x * TARGET_WIDTH, k.y * new_h]
                    ankl = [a.x * TARGET_WIDTH, a.y * new_h]
                    shld = [s.x * TARGET_WIDTH, s.y * new_h]
                    
                    knee_angle = calculate_angle(hip, knee, ankl)
                    
                    if knee_angle < 140 and stage == "ylhaalla":
                        stage = "alhaalla"
                        min_angle = 180
                    if stage == "alhaalla":
                        min_angle = min(min_angle, knee_angle)
                        if knee_angle > 155:
                            stage = "ylhaalla"
                            counter += 1
                            rep_history.append(int(min_angle))

                    # Piirretään grafiikat
                    cv2.line(frame, (int(shld[0]), int(shld[1])), (int(hip[0]), int(hip[1])), (0, 255, 0), 3)
                    cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 255), 2)
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), (0, 255, 0), 3)
                    cv2.putText(frame, f"REPS: {counter} Angle: {int(knee_angle)}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Kirjoitetaan frame videotiedostoon PyAV:lla
            av_frame = av.VideoFrame.from_ndarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

            processed_count += 1
            progress_bar.progress(min(1.0, target_frame_index / total_frames))

        # Viimeistellään videotiedosto
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    cap.release()
    st.success(f"Analyysi valmis! Toistot: {counter}")

    # NÄYTETÄÄN TULOKSET
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Analysoitu video")
        with open(output_path, "rb") as v_file:
            st.video(v_file.read())
    
    with col2:
        st.subheader("Toistojen syvyydet")
        for i, angle in enumerate(rep_history):
            st.write(f"Toisto {i+1}: **{angle}°**")
            
    # Poistetaan väliaikaistiedosto
    if os.path.exists(output_path):
        os.remove(output_path)
