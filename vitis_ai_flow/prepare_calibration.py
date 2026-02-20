import os
import shutil
import random

# ===============================
# CONFIG
# ===============================
SRC_IMG_DIR = "coco128/images/train2017"
SRC_LBL_DIR = "coco128/labels/train2017"

CALIB_DIR = "calibration"
IMG_DST = os.path.join(CALIB_DIR, "images")
LBL_DST = os.path.join(CALIB_DIR, "labels")

NUM_CALIB = 80

# ===============================
# CREATE FOLDERS
# ===============================
os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

print("Calibration folders created")

# ===============================
# SELECT RANDOM IMAGES
# ===============================
all_images = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith(".jpg")]

random.seed(0)
selected_images = random.sample(all_images, NUM_CALIB)

# ===============================
# COPY IMAGES + LABELS
# ===============================
for img in selected_images:
    lbl = img.replace(".jpg", ".txt")

    shutil.copy(os.path.join(SRC_IMG_DIR, img),
                os.path.join(IMG_DST, img))

    shutil.copy(os.path.join(SRC_LBL_DIR, lbl),
                os.path.join(LBL_DST, lbl))

print("Calibration dataset prepared successfully.")
