# isl_realtime_knn.py (ISL → gTTS with 2s stable detection + green guide box)
import cv2, mediapipe as mp, numpy as np, joblib, time, warnings, threading, os, uuid
from queue import Queue
from gtts import gTTS
from playsound import playsound

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ---------------------------
# Load KNN model
# ---------------------------
MODEL_PATH = "voice_to_isl/knn_artifacts/artifacts.pkl"
art = joblib.load(MODEL_PATH)
scaler, pca, knn, labels = art['scaler'], art['pca'], art['knn'], art['labels']

# ---------------------------
# Thread-safe TTS (using gTTS)
# ---------------------------
tts_queue = Queue()

def tts_worker():
    """Persistent TTS thread that speaks every item in the queue."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            # unique filename per request
            filename = f"temp_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("TTS Error:", e)
        tts_queue.task_done()

# Start worker thread
threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    """Add symbol to TTS queue immediately."""
    tts_queue.put(str(text))

# ---------------------------
# Normalize landmarks
# ---------------------------
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

# ---------------------------
# Mediapipe setup
# ---------------------------
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# ---------------------------
# Stable detection control
# ---------------------------
stable_symbol = None
stable_start_time = None
HOLD_TIME = 2.0  # seconds required to confirm a symbol

# ---------------------------
# Main loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw green guide box in center
    h, w, _ = frame.shape
    box_size = 300
    x1, y1 = w // 2 - box_size // 2, h // 2 - box_size // 2
    x2, y2 = w // 2 + box_size // 2, h // 2 + box_size // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

        # Show live prediction
        cv2.putText(frame, f"{pred}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Stable detection logic
        if stable_symbol == pred:
            if time.time() - stable_start_time >= HOLD_TIME:
                print("Confirmed:", pred)
                speak(pred)
                stable_symbol = None
                stable_start_time = None
        else:
            stable_symbol = pred
            stable_start_time = time.time()

    cv2.imshow("ISL Recognition (KNN)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)  # stop TTS thread
