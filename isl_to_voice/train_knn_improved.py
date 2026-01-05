
# train_knn_twohand.py
import os, cv2, joblib, random
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# -------------------
# PATHS
# -------------------
DATASET_DIR = r"voice_to_isl/memA_dataset/data"
OUT_DIR = "knn_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------
# CONFIG
# -------------------
SAMPLES_PER_CLASS = 200
ADD_MIRROR = True
PCA_COMPONENTS = 40
K = 3

# -------------------
# Mediapipe setup
# -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

def normalize_landmarks(coords):
    coords = coords.copy().astype(np.float32)
    wrist = coords[0, :2].copy()
    coords[:, :2] -= wrist
    dists = np.linalg.norm(coords[:, :2], axis=1)
    hand_size = dists.max()
    if hand_size <= 1e-6:
        hand_size = 1.0
    coords[:, :2] /= hand_size
    coords[:, 2] -= coords[0, 2]
    coords[:, 2] /= hand_size
    xy = coords[:, :2].flatten()
    z = coords[:, 2].flatten()
    return np.concatenate([xy, z])  # length 63

def mirror_vector(vec):
    vec = vec.copy()
    x_indices = np.arange(0, 21*2, 2)
    vec[x_indices] = -vec[x_indices]
    return vec

def extract_features(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    all_vecs = []
    for lm in res.multi_hand_landmarks:
        coords = np.array([[pt.x, pt.y, pt.z] for pt in lm.landmark])
        all_vecs.append(normalize_landmarks(coords))
    if len(all_vecs) == 1:
        feat = np.concatenate([all_vecs[0], np.zeros(63)])  # pad to 126
    else:
        feat = np.concatenate(all_vecs[:2])  # take two hands = 126
    return feat

# -------------------
# Load dataset
# -------------------
labels = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print("Found labels:", labels)

X, y = [], []

for label in labels:
    folder = os.path.join(DATASET_DIR, label)
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if not imgs:
        print("No images for", label); continue
    if SAMPLES_PER_CLASS and len(imgs) > SAMPLES_PER_CLASS:
        imgs = random.sample(imgs, SAMPLES_PER_CLASS)

    for img_path in tqdm(imgs, desc=f"Processing {label}"):
        img = cv2.imread(img_path)
        if img is None: continue
        feat = extract_features(img)
        if feat is None: continue
        X.append(feat); y.append(label)
        if ADD_MIRROR:
            X.append(mirror_vector(feat[:63]).tolist() + feat[63:].tolist()); y.append(label)

X = np.array(X); y = np.array(y)
print("Total samples:", X.shape)

# -------------------
# Split + Preprocess
# -------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

if PCA_COMPONENTS > 0:
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
else:
    pca = None
    X_train_p, X_val_p = X_train_s, X_val_s

# -------------------
# Train KNN
# -------------------
knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean", weights="distance", n_jobs=-1)
knn.fit(X_train_p, y_train)

print(f"Train acc: {knn.score(X_train_p, y_train):.4f}, Val acc: {knn.score(X_val_p, y_val):.4f}")

# -------------------
# Save artifacts
# -------------------
art = {'scaler': scaler, 'pca': pca, 'knn': knn, 'labels': labels}
joblib.dump(art, os.path.join(OUT_DIR, 'artifacts.pkl'), compress=3)
print("Saved artifacts to", os.path.join(OUT_DIR, 'artifacts.pkl'))