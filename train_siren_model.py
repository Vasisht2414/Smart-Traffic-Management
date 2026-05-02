import os
import librosa
import numpy as np
import torch
import torch.nn as nn

# ======================================================
# CONFIG
# ======================================================
SR = 22050
N_MFCC = 40
MAX_LEN = 40

DATASET_PATH = "ADD THE FILE LOCATION"
CLASSES = ["traffic", "ambulance", "firetruck"]
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_mfcc(path):
    y, sr = librosa.load(path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    mfcc = mfcc[:, :MAX_LEN]
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_LEN - mfcc.shape[1])),
            mode="constant"
        )
    return mfcc

# ======================================================
# LOAD DATASET (SAFE FILE FILTERING)
# ======================================================
X, y = [], []

for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_PATH, cls)

    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    for file in os.listdir(folder):
        # ---------- FILE FILTER ----------
        if not file.lower().endswith(AUDIO_EXTENSIONS):
            continue

        path = os.path.join(folder, file)

        try:
            mfcc = extract_mfcc(path)
            X.append(mfcc)
            y.append(label)
        except Exception as e:
            print(f"❌ Skipped file: {file} | Reason: {e}")

print(f"✅ Loaded {len(X)} audio samples")

# ======================================================
# CONVERT TO TENSORS
# ======================================================
X = torch.tensor(np.array(X)).unsqueeze(1).float()
y = torch.tensor(y)

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
            nn.Linear(64, 3)  # 3 classes
        )

    def forward(self, x):
        return self.net(x)

model = SirenNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# ======================================================
# TRAINING
# ======================================================
EPOCHS = 20

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

# ======================================================
# SAVE MODEL
# ======================================================
torch.save(model.state_dict(), "siren_model.pth")
print("🎉 Siren model trained and saved as siren_model.pth")
