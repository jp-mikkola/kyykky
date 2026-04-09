import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import av
import os

# --- ASETUKSET ---
st.set_page_config(page_title="AI Kyykkyanalyysi", layout="wide")
st.title("🏋️ AI Kyykkyanalyysi - Vakaa versio")

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
    tfile.close() # Suljetaan, jotta OpenCV voi lukea sen
    
    output_path = "output_analyzed.mp4"
    if os.path.exists(output_path):
        os.remove(output_path)
    
    cap = cv2.VideoCapture(tfile.name)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, int(original_fps / TARGET_FPS))
    
    st.info("Analysoidaan videota... Tämä vaihe kestää hetken.")
    progress_bar = st.progress(0)
    
    # Haetaan ekasta framesta mitat parillisina
    ret_first, first_frame = cap.read()
    if not ret_first:
        st.error("Videota ei voitu lukea.")
        st.stop()
        
    h_orig, w_orig, _ = first_frame.shape
    new_w = TARGET_WIDTH - (TARGET_WIDTH % 2) # Pakota parilliseksi
    new_h = int(new_w * (h_orig / w_orig))
    new_h = new_h - (new_h % 2) # Pakota parilliseksi

    # Alustetaan PyAV
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=TARGET_FPS)
    stream.width = new_w
    stream.height = new_h
    stream.pix_fmt = "yuv420p"
    # Lisätään optioita vakauden parantamiseksi
    stream.options = {"preset": "veryfast", "crf": "23"}
    
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
        # Palautetaan lukupää alkuun
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while cap.isOpened():
            target_frame_index = processed_count * skip_frames
            if target_frame_index >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (new_w, new_h))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((target_frame_index / original_fps) * 1000)
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                for landmarks in results.pose_landmarks:
                    # Pisteet
                    s = landmarks[11]; h_pt = landmarks[23]; k = landmarks[25]; a = landmarks[27]
                    shld = [s.x * new_w, s.y * new_h]; hip = [h_pt.x * new_w, h_pt.y * new_h]
                    knee = [k.x * new_w, k.y * new_h]; ankl = [a.x * new_w, a.y * new_h]
                    
                    knee_angle = calculate_angle(hip, knee, ankl)
                    
                    if knee_angle < 140 and stage == "ylhaalla":
                        stage = "alhaalla"; min_angle = 180
                    if stage == "alhaalla":
                        min_angle = min(min_angle, knee_angle)
                        if knee_angle > 155:
                            stage = "ylhaalla"
                            counter += 1
                            rep_history.append(int(min_angle))

                    # Grafiikat
                    cv2.line(frame, (int(shld[0]), int(shld[1])), (int(hip[0]), int(hip[1])), (0, 255, 0), 3)
                    cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 255), 2)
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), (0, 255, 0), 3)
                    cv2.putText(frame, f"REPS: {counter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Tallennetaan frame PyAV:lla
            # Muutetaan BGR (OpenCV) takaisin RGB:ksi videota varten
            vid_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(vid_frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

            processed_count += 1
            progress_bar.progress(min(1.0, target_frame_index / total_frames))

        # Suljetaan streamit
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    cap.release()
    st.success(f"Analyysi valmis! Toistot: {counter}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Analysoitu video")
        with open(output_path, "rb") as v_file:
            st.video(v_file.read())
    
    with col2:
        st.subheader("Toistojen syvyydet")
        for i, angle in enumerate(rep_history):
            st.write(f"Toisto {i+1}: **{angle}°**")
    
    # Siivotaan väliaikaistiedostot
    os.unlink(tfile.name)
