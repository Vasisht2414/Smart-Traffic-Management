import sounddevice as sd
import librosa
import torch
import numpy as np
import time
import torch.nn as nn
from collections import deque

# ======================================================
# CONFIG (CALIBRATED FOR YOUR MIC)
# ======================================================
SR = 22050
N_MFCC = 40
MAX_LEN = 40

PROB_THRESHOLD = 0.5           # relaxed ML confidence
BAND_ENERGY_THRESHOLD = 0.004  # phone speaker friendly
MIN_RMS = 0.002                # <<< CRITICAL FIX
MODULATION_THRESHOLD = 0.002   # allows phone siren

BUFFER_SIZE = 10
VOTE_THRESHOLD = 6
ACTIVE_TIME = 6.0

# ======================================================
# MODEL
# ======================================================
class SirenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# ======================================================
# LOAD MODEL
# ======================================================
model = SirenNet()
model.load_state_dict(torch.load("siren_model.pth", map_location="cpu"))
model.eval()

# ======================================================
# GLOBAL STATE
# ======================================================
siren_detected = False
last_detected = -9999
pred_buffer = deque(maxlen=BUFFER_SIZE)

# ======================================================
# AUDIO FEATURES
# ======================================================
def rms_energy(y):
    return np.sqrt(np.mean(y ** 2))

def band_energy(y):
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    band = spectrum[(freqs > 400) & (freqs < 2000)]
    return np.mean(band) if len(band) else 0.0

def modulation_score(y):
    env = np.abs(y)
    env = env - np.mean(env)
    return np.std(env)

# ======================================================
# LISTENER THREAD
# ======================================================
def listen():
    global siren_detected, last_detected

    start_time = time.time()

    while True:
        # Mic warm-up
        if time.time() - start_time < 3:
            siren_detected = False
            time.sleep(0.1)
            continue

        audio = sd.rec(int(0.5 * SR),
                        samplerate=SR,
                        channels=1,
                        dtype="float32")
        sd.wait()
        y = audio.flatten()

        # ---------- HARD FILTERS ----------
        if rms_energy(y) < MIN_RMS:
            siren_detected = (time.time() - last_detected) < ACTIVE_TIME
            continue

        if band_energy(y) < BAND_ENERGY_THRESHOLD:
            siren_detected = (time.time() - last_detected) < ACTIVE_TIME
            continue

        if modulation_score(y) < MODULATION_THRESHOLD:
            siren_detected = (time.time() - last_detected) < ACTIVE_TIME
            continue

        # ---------- ML CHECK ----------
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
        mfcc = mfcc[:, :MAX_LEN]
        if mfcc.shape[1] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])))

        x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)[0]
            pred = torch.argmax(probs).item()

        if pred in [1, 2] and probs[pred] > PROB_THRESHOLD:
            last_detected = time.time()

        siren_detected = (time.time() - last_detected) < ACTIVE_TIME
