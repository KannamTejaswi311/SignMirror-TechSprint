# integrated_app.py
import streamlit as st
import os
import re
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import warnings
import threading
from queue import Queue
from gtts import gTTS
from playsound import playsound
import whisper
import yaml
import imageio
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ========================
# Configs
# ========================
MEMB_A_MODEL_PATH = "voice_to_isl/knn_artifacts/artifacts.pkl"
MEMB_B_MODEL_NAME = "base"
MEMB_B_ANIMATIONS_PATH = "voice_to_isl/animations"
MEMB_B_TOKENS_FILE = "voice_to_isl/tokens.yaml"

# ========================
# Member A (Sign → Voice)
# ========================
tts_queue = Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            filename = f"temp_{np.random.randint(1e6)}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("TTS Error:", e)
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    tts_queue.put(str(text))

def normalize_landmarks(coords):
    coords = coords.copy().astype(np.float32)
    wrist = coords[0, :2].copy()
    coords[:, :2] -= wrist
    dists = np.linalg.norm(coords[:, :2], axis=1)
    hand_size = dists.max() if dists.max() > 1e-6 else 1.0
    coords[:, :2] /= hand_size
    coords[:, 2] -= coords[0, 2]
    coords[:, 2] /= hand_size
    return np.concatenate([coords[:, :2].flatten(), coords[:, 2].flatten()])

def run_sign_to_voice_app(placeholder):
    placeholder.write("👐 Deaf → Hearing (Sign → Voice)")

    art = joblib.load(MEMB_A_MODEL_PATH)
    scaler, pca, knn, labels = art['scaler'], art['pca'], art['knn'], art['labels']

    mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    frame_placeholder = placeholder.empty()
    stable_symbol, stable_start_time = None, None
    HOLD_TIME = 2.0  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            placeholder.warning("Camera not accessible")
            break

        # Green guide box
        h, w, _ = frame.shape
        box_size = 300
        x1, y1 = w//2 - box_size//2, h//2 - box_size//2
        x2, y2 = w//2 + box_size//2, h//2 + box_size//2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            feats = []
            for lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                coords = np.array([[pt.x, pt.y, pt.z] for pt in lm.landmark])
                feats.append(normalize_landmarks(coords))

            if len(feats) == 1:
                feature_vec = np.concatenate([feats[0], np.zeros(63)])
            else:
                feature_vec = np.concatenate(feats[:2])

            feature_vec = feature_vec.reshape(1, -1)
            feature_vec = scaler.transform(feature_vec)
            if pca is not None:
                feature_vec = pca.transform(feature_vec)

            pred = knn.predict(feature_vec)[0]

            # Stable detection
            if stable_symbol == pred:
                if time.time() - stable_start_time >= HOLD_TIME:
                    speak(pred)
                    stable_symbol, stable_start_time = None, None
            else:
                stable_symbol, stable_start_time = pred, time.time()

            cv2.putText(frame, f"{pred}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

# ========================
# Member B (Voice → ISL)
# ========================
def find_gif_path(gloss):
    candidates = [f"{gloss}.gif", f"{gloss.upper()}.gif", f"{gloss.lower()}.gif"]
    for c in candidates:
        p = os.path.join(MEMB_B_ANIMATIONS_PATH, c)
        if os.path.exists(p):
            return p
    return None

def gif_to_mp4(gif_path):
    temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    gif = imageio.mimread(gif_path)
    frames_rgb = []
    for frame in gif:
        if frame.shape[-1] == 4:
            frame_rgb = frame[..., :3]
        else:
            frame_rgb = frame
        frames_rgb.append(frame_rgb)
    imageio.mimsave(temp_mp4.name, frames_rgb, fps=10)
    return temp_mp4.name

def play_animation_streamlit(gloss, placeholder, speed_factor=1.0):
    p = find_gif_path(gloss)
    if not p:
        placeholder.warning(f"No GIF found for: {gloss}")
        return
    delay = (1.0 if len(gloss)==1 else 3.0)*speed_factor
    if len(gloss) == 1:
        placeholder.image(p, width=150)
    else:
        mp4_path = gif_to_mp4(p)
        placeholder.video(mp4_path)
    time.sleep(delay)

def play_sequence_streamlit(seq, placeholder, speed_factor=1.0):
    for gloss in seq:
        play_animation_streamlit(gloss, placeholder, speed_factor)

def text_to_gloss_sequence(text, tokens_map):
    clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = clean.split()
    seq = []
    for word in words:
        if find_gif_path(word):
            seq.append(word)
        else:
            for ch in word:
                if find_gif_path(ch.upper()):
                    seq.append(ch.upper())
                elif find_gif_path(ch.lower()):
                    seq.append(ch.lower())
                elif ch.isdigit() and find_gif_path(ch):
                    seq.append(ch)
    return seq

def update_tokens_from_gifs():
    if os.path.exists(MEMB_B_TOKENS_FILE):
        with open(MEMB_B_TOKENS_FILE, "r") as f:
            tokens_map = yaml.safe_load(f) or {}
    else:
        tokens_map = {}
    for f in os.listdir(MEMB_B_ANIMATIONS_PATH):
        if f.lower().endswith(".gif"):
            key = os.path.splitext(f)[0].lower()
            if key not in tokens_map:
                tokens_map[key] = key
    with open(MEMB_B_TOKENS_FILE, "w") as f:
        yaml.dump(tokens_map, f, sort_keys=True)
    return tokens_map

def record_microphone_audio(duration=5, fs=48000, filename="temp_mic.wav"):
    st.info(f"Recording {duration}s...")
    audio = sd.rec(int(duration*fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, np.int16(audio*32767))
    st.success(f"Saved {filename}")
    return filename

@st.cache_resource
def load_whisper_model():
    return whisper.load_model(MEMB_B_MODEL_NAME)

def run_voice_to_sign_app(placeholder):
    placeholder.write("🎤 Hearing → Deaf (Voice → ISL)")
    model = load_whisper_model()
    tokens_map = update_tokens_from_gifs()
    animation_placeholder = placeholder.empty()

    option = placeholder.radio("Input method:", ["Upload Audio File","Use Microphone"])
    audio_path = None

    if option=="Upload Audio File":
        uploaded_file = placeholder.file_uploader("Upload Audio file", type=["wav","mp3","m4a"])
        if uploaded_file is not None:
            temp_file = "temp_audio.wav"
            with open(temp_file,"wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_path = temp_file
    else:
        duration = placeholder.number_input("Recording duration (seconds)", 1,30,5)
        if placeholder.button("🔴 Record"):
            audio_path = record_microphone_audio(duration)

    if audio_path:
        placeholder.info("Transcribing...")
        result = model.transcribe(audio_path, fp16=False)
        transcribed_text = result["text"].strip()
        placeholder.subheader("Transcription:")
        placeholder.write(transcribed_text)

        isl_sequence = text_to_gloss_sequence(transcribed_text, tokens_map)
        placeholder.subheader("ISL Sequence:")
        placeholder.write(isl_sequence)

        placeholder.subheader("Animation Playback")
        play_sequence_streamlit(isl_sequence, animation_placeholder, speed_factor=1.8)

# ========================
# Streamlit Interface
# ========================
st.set_page_config(layout="wide")
st.title("👐 SignMirror: Deaf & Hearing Communication")

col1, col2 = st.columns(2)

with col1:
    run_voice_to_sign_app(st)

with col2:
    run_sign_to_voice_app(st)