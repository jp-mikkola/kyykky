import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import av
import time

# --- SIVUN ASETUKSET ---
st.set_page_config(page_title="AI Kyykkyvalmentaja", layout="wide")
st.title("🏋️ Live AI Kyykkyvalmentaja")
st.write("Aseta puhelin sivulle ja paina 'Start'.")

# --- MEDIAPIPE SETUP ---
MODEL_PATH = 'pose_landmarker_lite.task'

# --- APUFUNKTIOT ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def calculate_vertical_angle(top, bottom):
    radians = np.arctan2(top[0] - bottom[0], bottom[1] - top[1])
    return radians * 180.0 / np.pi

# --- VIDEO-PROSESSORI ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # Alustetaan MediaPipe Tasks
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        
        # Huomio: Kaikkien alla olevien rivien pitää olla samalla tasolla (sennys)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # TÄMÄ RIVI aiheutti virheen, jos se oli eri kohdassa kuin 'options' yläpuolella
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        
        # Treenimuuttujat
        self.counter = 0
        self.stage = "ylhaalla"
        self.min_knee_angle = 180
        self.last_feedback = "Valmiina!"

    def transform(self, frame):
        # Muunnetaan WebRTC-frame numpy-taulukoksi (BGR)
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Peilikuva mobiilikäyttöön
        img = cv2.flip(img, 1)
        
        # MediaPipe vaatii RGB ja aikaleiman
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        timestamp = int(time.time() * 1000)
        
        results = self.landmarker.detect_for_video(mp_image, timestamp)

        if results.pose_landmarks:
            for landmarks in results.pose_landmarks:
                # Valitaan puoli (tässä yksinkertaistettu vasen)
                s = landmarks[11]; h_pt = landmarks[23]; k = landmarks[25]; a = landmarks[27]; e = landmarks[7]
                
                shld = [s.x * w, s.y * h]
                hip = [h_pt.x * w, h_pt.y * h]
                knee = [k.x * w, k.y * h]
                ankl = [a.x * w, a.y * h]
                ear = [e.x * w, e.y * h]

                # Laskenta
                knee_angle = calculate_angle(hip, knee, ankl)
                back_angle = calculate_vertical_angle(shld, hip)
                shin_angle = calculate_vertical_angle(knee, ankl)
                
                # Kulmaerot (suunnalla)
                lean_diff = abs(back_angle - shin_angle)
                
                # Toistolaskuri
                if knee_angle < 140 and self.stage == "ylhaalla":
                    self.stage = "alhaalla"
                    self.min_knee_angle = 180
                
                if self.stage == "alhaalla":
                    self.min_knee_angle = min(self.min_knee_angle, knee_angle)
                    if knee_angle > 155:
                        self.stage = "ylhaalla"
                        if self.min_knee_angle < 140:
                            self.counter += 1
                            self.last_feedback = f"Rep {self.counter} OK! Syvyys: {int(self.min_knee_angle)}°"

                # Visualisointi suoraan kuvaan
                color = (0, 0, 255) if lean_diff > 15 else (0, 255, 0)
                cv2.line(img, (int(shld[0]), int(shld[1])), (int(hip[0]), int(hip[1])), color, 4)
                cv2.line(img, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 255), 2)
                cv2.line(img, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), color, 4)

                # Overlay-tekstit (Musta tausta tekstille parantaa luettavuutta)
                cv2.rectangle(img, (0,0), (350, 150), (0,0,0), -1)
                cv2.putText(img, f"TOISTOT: {self.counter}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f"POLVI: {int(knee_angle)} deg", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, self.last_feedback, (20, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # PALAUTUS: Tämä on se kriittinen osa joka puuttui!
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- KÄYNNISTYS ---
webrtc_streamer(
    key="squat-coach",
    video_frame_callback=None, # Käytämme transformer-luokkaa
    video_transformer_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}, # Vain video
)
