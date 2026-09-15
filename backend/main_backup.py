from pathlib import Path
import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.signal as signal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

# Our trained model is currently inside:
# ecg-noise-analyzer/ml/models/
ML_DIR = BASE.parent / "ml"
MODEL_DIR = ML_DIR / "models"

MODEL_PATH = MODEL_DIR / "paper_1d_cnn_noise.pt"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ECG Noise Analyzer API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PAPER 1D CNN
# ============================================================

class Paper1DCNN(nn.Module):
    """
    Binary clinical-noise classifier reproduced from the notebook.
    
    Input:
        (batch, 1, 1000)

    Output:
        2 classes
        0 = Clean
        1 = Noisy
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv1d(
            1, 32, 3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(
            32, 48, 3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(48)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(
            48, 64, 2
        )
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool1d(2, 2)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(64, 64)
        self.fc_relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(64, 32)
        self.fc_relu2 = nn.LeakyReLU()

        self.classifier = nn.Linear(
            32, num_classes
        )

    def forward(self, x):

        x = self.pool1(
            self.relu1(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool2(
            self.relu2(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.pool3(
            self.relu3(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        x = self.global_pool(x).squeeze(-1)

        x = self.fc_relu1(
            self.fc1(x)
        )

        x = self.fc_relu2(
            self.fc2(x)
        )

        return self.classifier(x)


# ============================================================
# LOAD MODEL
# ============================================================

def load_1d_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = Paper1DCNN(num_classes=2)

    state = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state)

    model.eval()

    return model


# ============================================================
# READ ECG FILE
# ============================================================

def read_numeric_upload(raw: bytes) -> np.ndarray:

    try:

        text = raw.decode(
            "utf-8",
            errors="ignore"
        )

        # Try CSV-style parsing first
        df = pd.read_csv(
            io.StringIO(text),
            header=None
        )

        values = (
            df.apply(
                pd.to_numeric,
                errors="coerce"
            )
            .to_numpy()
            .reshape(-1)
        )

        values = values[
            np.isfinite(values)
        ]

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Could not parse ECG file: {exc}"
        )

    if values.size == 0:

        raise HTTPException(
            status_code=400,
            detail="No numeric ECG samples found."
        )

    return values.astype(
        np.float32
    )


# ============================================================
# PREPROCESS ECG
# ============================================================

TARGET_SAMPLES = 1000
TARGET_FS = 200
ORIGINAL_FS = 360


def prepare_1d_signal(
    values: np.ndarray
) -> np.ndarray:

    """
    Prepare uploaded ECG for the trained model.

    The notebook's clinical-noise pipeline uses:

        Original sampling rate = 360 Hz
        Target sampling rate   = 200 Hz
        Window length          = 5 seconds
        Samples                = 1000

    Therefore the CNN expects:

        (1, 1000)
    """

    if len(values) == 0:

        raise HTTPException(
            status_code=400,
            detail="ECG signal is empty."
        )

    # --------------------------------------------------------
    # If signal is longer than 1000 samples:
    # take the first 1000 samples after resampling.
    #
    # If signal has fewer than 1000 samples:
    # resample it directly to 1000.
    # --------------------------------------------------------

    if len(values) != TARGET_SAMPLES:

        values = signal.resample(
            values,
            TARGET_SAMPLES
        )

    return values.astype(
        np.float32
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",

        "model": {
            "name": "Paper 1D CNN",
            "task": "Clinical ECG Noise Classification",
            "classes": {
                "0": "Clean",
                "1": "Noisy"
            },
            "input_samples": 1000,
            "sampling_rate": 200,
            "weights_available": MODEL_PATH.exists()
        }
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("1d_cnn")
):

    # --------------------------------------------------------
    # Check model selection
    # --------------------------------------------------------

    if model != "1d_cnn":

        raise HTTPException(
            status_code=501,
            detail=(
                "Only the trained 1D CNN is currently "
                "available. ResNet18 inference will be "
                "added after its trained checkpoint is available."
            )
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    raw = await file.read()

    if not raw:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    values = read_numeric_upload(raw)

    original_samples = len(values)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    try:

        net = load_1d_model()

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not load trained model: {exc}"
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    x = prepare_1d_signal(values)

    # Model expects:
    #
    # (batch, channels, samples)
    #
    # (1, 1, 1000)

    tensor = torch.from_numpy(
        x
    ).view(
        1,
        1,
        TARGET_SAMPLES
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = net(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )

    # --------------------------------------------------------
    # Convert class to meaningful label
    # --------------------------------------------------------

    class_names = {
        0: "Clean",
        1: "Noisy"
    }

    label = class_names[
        predicted_class
    ]

    confidence = float(
        probabilities[
            predicted_class
        ].item()
    )

    clean_probability = float(
        probabilities[0].item()
    )

    noisy_probability = float(
        probabilities[1].item()
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "success": True,

        "model": "1d_cnn",

        "prediction": {
            "class_id": predicted_class,
            "label": label,
            "confidence": confidence
        },

        "probabilities": {
            "clean": clean_probability,
            "noisy": noisy_probability
        },

        "signal": {
            "samples_received": original_samples,
            "samples_used": TARGET_SAMPLES,
            "target_sampling_rate": TARGET_FS
        },

        "message": (
            "ECG signal analyzed using the trained "
            "Paper 1D CNN clinical-noise classifier."
        )
    }