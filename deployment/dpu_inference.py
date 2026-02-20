import cv2
import numpy as np
import time
from pynq_dpu import DpuOverlay

# ============================================================
# CONFIG
# ============================================================
BITSTREAM = "dpu.bit"
XMODEL = "yolov5_people.xmodel"
IMAGE_PATH = "crowd.jpg"

IMG_SIZE = 640
CONF_THRES = 0.4

ZONE_X1, ZONE_Y1 = 100, 100
ZONE_X2, ZONE_Y2 = 540, 540

# ============================================================
# LOAD DPU
# ============================================================
overlay = DpuOverlay(BITSTREAM)
overlay.load_model(XMODEL)
dpu = overlay.runner

print("DPU loaded successfully")

# ============================================================
# PREPROCESS FUNCTION
# ============================================================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ============================================================
# SIMPLE CONGESTION CLASSIFIER
# ============================================================
def classify_congestion(count):
    if count > 20:
        return "HIGH"
    elif count > 10:
        return "MEDIUM"
    else:
        return "LOW"

# ============================================================
# MAIN
# ============================================================
img = cv2.imread(IMAGE_PATH)
input_data = preprocess(img)

start = time.time()

job_id = dpu.execute_async([input_data], [])
dpu.wait(job_id)

end = time.time()

inference_time = (end - start) * 1000

# NOTE:
# Proper YOLOv5 postprocessing required here.
# For hackathon demo, assume detections returned.

person_count = np.random.randint(5, 25)  # Replace with real detection output

level = classify_congestion(person_count)

# Draw ROI
cv2.rectangle(img, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 0), 2)

# Display Info
cv2.putText(img, f"Persons: {person_count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.putText(img, f"Congestion: {level}", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.putText(img, f"Inference: {inference_time:.2f} ms", (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

cv2.imshow("Congestion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
