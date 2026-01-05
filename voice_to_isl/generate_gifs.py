import os
import imageio

# ----------------------------
# Paths
# ----------------------------
MEM_A_DATASET = os.path.join(os.path.dirname(__file__), "memA_dataset", "data")  # notice /data
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "animations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Generate GIFs
# ----------------------------
for label in os.listdir(MEM_A_DATASET):
    label_path = os.path.join(MEM_A_DATASET, label)
    if not os.path.isdir(label_path):
        continue

    images = []
    for img_file in sorted(os.listdir(label_path)):
        if img_file.lower().endswith(".jpg"):
            images.append(imageio.imread(os.path.join(label_path, img_file)))

    if images:
        gif_path = os.path.join(OUTPUT_DIR, f"{label}.gif")
        imageio.mimsave(gif_path, images, duration=0.1)
        print(f"✅ Saved GIF: {gif_path}")
    else:
        print(f"⚠️ No images found for label: {label}")
