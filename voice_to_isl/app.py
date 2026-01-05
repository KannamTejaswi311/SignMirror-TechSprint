# voice_to_isl/app.py
import os, re, yaml, whisper, imageio, tempfile, time
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import functools

# ========================
# Config
# ========================
MODEL_NAME = "base"
ANIMATIONS_PATH = "animations"
TOKENS_FILE = "tokens.yaml"

# ========================
# Whisper Loader
# ========================
@functools.lru_cache(maxsize=1)
def load_whisper_model(model_size: str = MODEL_NAME):
    """
    Loads and caches the Whisper model so it can be reused
    by both local and networked apps (like role_based_room_app.py).
    """
    model = whisper.load_model(model_size)
    return model

# ========================
# Helpers
# ========================
def find_gif_path(gloss: str):
    candidates = [f"{gloss}.gif", f"{gloss.upper()}.gif", f"{gloss.lower()}.gif"]
    for c in candidates:
        p = os.path.join(ANIMATIONS_PATH, c)
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

def play_animation_streamlit(gloss: str, placeholder, speed_factor=1.0):
    p = find_gif_path(gloss)
    if not p:
        st.warning(f"⚠ No ISL animation found for: {gloss}")
        return
    delay = (1.0 if len(gloss) == 1 else 3.0) * speed_factor
    if len(gloss) == 1:
        placeholder.image(p, width=150)
    else:
        mp4_path = gif_to_mp4(p)
        placeholder.video(mp4_path, start_time=0)
    time.sleep(delay)

def play_sequence_streamlit(seq, placeholder, speed_factor=1.0):
    for gloss in seq:
        play_animation_streamlit(gloss, placeholder, speed_factor)

def text_to_gloss_sequence(text: str, tokens_map: dict):
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

def update_tokens_from_gifs(tokens_file=TOKENS_FILE, animations_path=ANIMATIONS_PATH):
    if os.path.exists(tokens_file):
        with open(tokens_file, "r") as f:
            tokens_map = yaml.safe_load(f) or {}
    else:
        tokens_map = {}
    for f in os.listdir(animations_path):
        if f.lower().endswith(".gif"):
            key = os.path.splitext(f)[0].lower()
            if key not in tokens_map:
                tokens_map[key] = key
    with open(tokens_file, "w") as f:
        yaml.dump(tokens_map, f, sort_keys=True)
    return tokens_map

def record_microphone_audio(duration=5, fs=48000, filename="temp_mic.wav"):
    st.info(f"🎙 Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, np.int16(audio * 32767))
    st.success(f"✅ Recording saved as {filename}")
    return filename

# ========================
# Main function for Streamlit
# ========================
def run_member_b():
    st.title("🎤 Voice to ISL Translator (Member B)")
    model = load_whisper_model(MODEL_NAME)
    tokens_map = update_tokens_from_gifs()
    animation_placeholder = st.empty()

    option = st.radio("Choose input method:", ["Upload Audio File", "Use Microphone"])
    audio_path = None

    if option == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
        if uploaded_file:
            temp_file = "temp_audio.wav"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_path = temp_file
            st.success("Audio uploaded successfully.")

    elif option == "Use Microphone":
        duration = st.number_input("Recording duration (seconds):", min_value=1, max_value=30, value=5)
        if st.button("🔴 Record Microphone"):
            audio_path = record_microphone_audio(duration=duration)

    if audio_path:
        st.info("🔄 Transcribing speech...")
        result = model.transcribe(audio_path, fp16=False)
        transcribed_text = result["text"].strip()
        st.subheader("✅ Transcription Result:")
        st.write(transcribed_text)

        isl_sequence = text_to_gloss_sequence(transcribed_text, tokens_map)
        st.subheader("🔗 ISL Sequence:")
        st.write(isl_sequence)

        st.subheader("🎞 ISL Animation Playback")
        play_sequence_streamlit(isl_sequence, animation_placeholder, speed_factor=1.8)
