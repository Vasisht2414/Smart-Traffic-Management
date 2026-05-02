import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from datetime import datetime
from collections import deque
import csv
import os

# ======================================================
# IMPORT SIREN MODULE
# ======================================================
import siren_listener_ml

# ======================================================
# INPUT SOURCE
# ======================================================
INPUT_SOURCE = 0  # webcam

# ======================================================
# LOAD YOLO MODELS (ENSEMBLE)
# ======================================================
custom_model = YOLO(
    "ADD THE YOLO MODEL LOCATION"
)
base_model = YOLO("yolov8n.pt")  # pretrained COCO

# ======================================================
# CONFIG
# ======================================================
CONF_CUSTOM = 0.15
CONF_BASE = 0.25

GREEN_LOW = 20
GREEN_MEDIUM = 35
GREEN_HIGH = 60
GREEN_EMERGENCY = 90

# ======================================================
# START SIREN LISTENER
# ======================================================
threading.Thread(
    target=siren_listener_ml.listen,
    daemon=True
).start()

# ======================================================
# STARTUP GRACE PERIOD
# ======================================================
program_start_time = time.time()
EMERGENCY_GRACE_TIME = 5

# ======================================================
# VEHICLE CLASSES
# ======================================================
CUSTOM_VEHICLES = [
    "car", "bus", "truck", "motorcycle",
    "bicycle", "auto_rickshaw", "e_rickshaw",
    "van", "pickup", "tractor", "trailer"
]

COCO_VEHICLES = ["car", "bus", "truck", "motorbike", "bicycle"]

# ======================================================
# TRAFFIC LEARNING (UNSUPERVISED)
# ======================================================
traffic_history = []
baseline_window = deque(maxlen=120)   # baseline learning
rush_hour_active = False

# ======================================================
# LEARNING LOG
# ======================================================
LOG_FILE = "traffic_learning.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Vehicle_Count", "Rush_Hour", "Mode"])

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def classify_traffic(count):
    if count < 15:
        return "LOW", GREEN_LOW
    elif count < 40:
        return "MEDIUM", GREEN_MEDIUM
    else:
        return "HIGH", GREEN_HIGH

def detect_rush_hour(count):
    global rush_hour_active
    if len(baseline_window) < 30:
        return False

    mean = np.mean(baseline_window)
    std = np.std(baseline_window)

    rush_hour_active = count > mean + 1.2 * std
    return rush_hour_active

def fallback_count():
    if rush_hour_active:
        return 45
    if traffic_history:
        return int(np.mean(traffic_history[-5:]))
    return 15

def log_learning(count, rush, mode):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            count,
            "YES" if rush else "NO",
            mode
        ])

# ======================================================
# VEHICLE DETECTION (ENSEMBLE + TRACKING)
# ======================================================
def detect_vehicles(frame):
    detections = []

    # -------- CUSTOM MODEL (TRACKING) --------
    res_custom = custom_model.track(
        frame, persist=True, conf=CONF_CUSTOM, iou=0.5, verbose=False
    )

    if res_custom and res_custom[0].boxes is not None:
        for b in res_custom[0].boxes:
            cls = custom_model.names[int(b.cls[0])]
            if cls in CUSTOM_VEHICLES:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                tid = int(b.id[0]) if b.id is not None else None
                detections.append((x1, y1, x2, y2, tid))

    # -------- FALLBACK TO PRETRAINED YOLO --------
    if len(detections) < 3:
        res_base = base_model.track(
            frame, persist=True, conf=CONF_BASE, iou=0.5, verbose=False
        )

        if res_base and res_base[0].boxes is not None:
            for b in res_base[0].boxes:
                cls = base_model.names[int(b.cls[0])]
                if cls in COCO_VEHICLES:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    area = w * h
                    aspect = w / max(h, 1)

                    # Roof / partial vehicle filter
                    if area > 800 and 0.6 < aspect < 2.8:
                        tid = int(b.id[0]) if b.id is not None else None
                        detections.append((x1, y1, x2, y2, tid))

    return detections

def count_vehicles(detections):
    ids = {d[4] for d in detections if d[4] is not None}
    return len(ids)

# ======================================================
# MAIN LOOP
# ======================================================
cap = cv2.VideoCapture(INPUT_SOURCE, cv2.CAP_DSHOW)
print("✅ Smart Traffic Management — FULL AI SYSTEM")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_vehicles(frame)
    count = count_vehicles(detections)

    if count > 0:
        mode = "VISION"
        baseline_window.append(count)
    else:
        count = fallback_count()
        mode = "FALLBACK"

    traffic_history.append(count)

    rush = detect_rush_hour(count)
    log_learning(count, rush, mode)

    # Emergency logic
    if time.time() - program_start_time < EMERGENCY_GRACE_TIME:
        emergency = False
    else:
        emergency = siren_listener_ml.siren_detected

    if emergency:
        traffic_state = "EMERGENCY"
        green = GREEN_EMERGENCY
    else:
        traffic_state, green = classify_traffic(count)

    # ==================================================
    # DRAW DETECTIONS
    # ==================================================
    for (x1, y1, x2, y2, tid) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        if tid is not None:
            cv2.putText(frame, f"ID:{tid}",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0), 1)

    # ==================================================
    # DISPLAY
    # ==================================================
    cv2.putText(frame, f"Vehicles: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(frame, f"Mode: {mode}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.putText(frame, f"Traffic: {traffic_state}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Green Time: {green}s", (20,160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.putText(frame,
                f"Rush Learned: {'YES' if rush else 'NO'}",
                (20,200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255,100,0), 2)

    if emergency:
        cv2.putText(frame,
                    "🚨 EMERGENCY VEHICLE DETECTED",
                    (20,240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)

    cv2.imshow("Smart Traffic Management — FINAL", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
