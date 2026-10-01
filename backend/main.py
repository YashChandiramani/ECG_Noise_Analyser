from pathlib import Path
import io

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import scipy.signal as signal

import gc

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision.models import resnet18

# Keep CPU inference single-threaded to prevent memory/thread bloat on Render
torch.set_num_threads(1)


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
ML_DIR = PROJECT_ROOT / "ml"
MODEL_DIR = ML_DIR / "models"
LEGACY_MODEL_DIR = BASE / "models"


def resolve_model_path(filename: str) -> Path:
    """Find a model anywhere the project may reasonably keep it."""

    candidates = [
        MODEL_DIR / filename,
        LEGACY_MODEL_DIR / filename,
        ML_DIR / filename,
        PROJECT_ROOT / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return candidates[0]


CNN_MODEL_PATH = resolve_model_path("paper_1d_cnn_noise.pt")
RESNET_MODEL_PATH = resolve_model_path("resnet18_noise.pt")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ECG Noise Analyzer API",
    version="1.0.0"
)

# Exception handling middleware so 500 errors still pass through CORSMiddleware
# with Access-Control-Allow-Origin headers attached, preventing browsers from
# masking internal errors with CORS policy violations.
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"}
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "ECG Noise Analyzer API",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================
# PAPER 1D CNN
# ============================================================

class Paper1DCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(32, 48, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(48)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(48, 64, 2)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool1d(2, 2)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(64, 64)
        self.fc_relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(64, 32)
        self.fc_relu2 = nn.LeakyReLU()

        self.classifier = nn.Linear(32, num_classes)

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
# LOAD 1D CNN
# ============================================================

_MODEL_1D = None
_MODEL_RESNET = None


def load_1d_model():

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            "1D CNN model not found. "
            f"Expected one of: {MODEL_DIR / 'paper_1d_cnn_noise.pt'}, "
            f"{LEGACY_MODEL_DIR / 'paper_1d_cnn_noise.pt'}, "
            f"{ML_DIR / 'paper_1d_cnn_noise.pt'}"
        )

    model = Paper1DCNN(num_classes=2)

    state = torch.load(
        CNN_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    del state
    gc.collect()

    return model


def get_1d_model():
    global _MODEL_1D
    if _MODEL_1D is None:
        _MODEL_1D = load_1d_model()
    return _MODEL_1D


# ============================================================
# RESNET-18
# ============================================================

def load_resnet_model():

    if not RESNET_MODEL_PATH.exists():
        raise FileNotFoundError(
            "ResNet-18 model not found. "
            f"Expected one of: {MODEL_DIR / 'resnet18_noise.pt'}, "
            f"{LEGACY_MODEL_DIR / 'resnet18_noise.pt'}, "
            f"{ML_DIR / 'resnet18_noise.pt'}"
        )

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    try:
        checkpoint = torch.load(
            RESNET_MODEL_PATH,
            map_location="cpu",
            weights_only=False,
            mmap=True
        )
    except Exception:
        checkpoint = torch.load(
            RESNET_MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, assign=True)
    except Exception:
        model.load_state_dict(state_dict)

    model.eval()

    del checkpoint
    del state_dict
    gc.collect()

    return model


def get_resnet_model():
    global _MODEL_RESNET
    if _MODEL_RESNET is None:
        _MODEL_RESNET = load_resnet_model()
    return _MODEL_RESNET


# ============================================================
# READ ECG FILE
# ============================================================

def read_numeric_upload(raw: bytes) -> np.ndarray:

    try:

        text = raw.decode(
            "utf-8",
            errors="ignore"
        )

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

    return values.astype(np.float32)


# ============================================================
# COMMON ECG SETTINGS
# ============================================================

TARGET_SAMPLES = 1000
TARGET_FS = 200
ORIGINAL_FS = 360
WINDOW_SEC = 5


# ============================================================
# 1D CNN PREPROCESSING
# ============================================================

def prepare_1d_signal(
    values: np.ndarray
) -> np.ndarray:

    if len(values) == 0:

        raise HTTPException(
            status_code=400,
            detail="ECG signal is empty."
        )

    if len(values) != TARGET_SAMPLES:

        values = signal.resample(
            values,
            TARGET_SAMPLES
        )

    return values.astype(
        np.float32
    )


# ============================================================
# RESNET PREPROCESSING
# ============================================================

def prepare_resnet_signal(
    values: np.ndarray
) -> np.ndarray:

    required_samples = ORIGINAL_FS * WINDOW_SEC

    if len(values) < required_samples:

        raise HTTPException(
            status_code=400,
            detail=(
                f"ResNet-18 requires at least "
                f"{required_samples} ECG samples "
                f"(5 seconds at 360 Hz). "
                f"Received {len(values)}."
            )
        )

    # Same 5-second window used by the
    # standalone ResNet inference script.
    values = values[:required_samples]

    # 360 Hz -> 200 Hz
    target_samples = TARGET_SAMPLES

    values = signal.resample(
        values,
        target_samples
    )

    return values.astype(np.float32)


# ============================================================
# CWT -> IMAGE
# ============================================================

def ecg_to_scalogram(
    ecg_signal: np.ndarray
) -> torch.Tensor:

    scales = np.arange(1, 129)

    coefficients, frequencies = pywt.cwt(
        ecg_signal,
        scales,
        "cmor1.5-1.0"
    )

    scalogram = np.abs(coefficients)

    scalogram = np.log1p(
        scalogram
    )

    min_value = scalogram.min()
    max_value = scalogram.max()

    if max_value > min_value:

        scalogram = (
            scalogram - min_value
        ) / (
            max_value - min_value
        )

    else:

        scalogram = np.zeros_like(
            scalogram
        )

    scalogram = (
        scalogram * 255
    ).astype(np.uint8)

    scalogram = torch.from_numpy(
        scalogram
    ).float()

    scalogram = scalogram.unsqueeze(0)

    scalogram = torch.nn.functional.interpolate(
        scalogram.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    scalogram = scalogram.squeeze()

    scalogram = scalogram.unsqueeze(0)

    # Grayscale -> RGB
    scalogram = scalogram.repeat(
        3,
        1,
        1
    )

    scalogram = scalogram / 255.0

    # IMPORTANT:
    # These are intentionally identical to
    # the standalone ResNet inference script.
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.406]
    ).view(3, 1, 1)

    scalogram = (
        scalogram - mean
    ) / std

    return scalogram


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "models": {

            "1d_cnn": {
                "name": "Paper 1D CNN",
                "task": "Clinical ECG Noise Classification",
                "classes": {
                    "0": "Clean",
                    "1": "Noisy"
                },
                "input_samples": 1000,
                "sampling_rate": 200,
                "weights_available": CNN_MODEL_PATH.exists()
            },

            "resnet18": {
                "name": "ResNet-18",
                "task": "Clinical ECG Noise Classification",
                "classes": {
                    "0": "Clean",
                    "1": "Noisy"
                },
                "input_size": [3, 224, 224],
                "sampling_rate": 200,
                "cwt_wavelet": "cmor1.5-1.0",
                "cwt_scales": [1, 128],
                "weights_available": RESNET_MODEL_PATH.exists()
            }
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

    # ========================================================
    # READ FILE
    # ========================================================

    raw = await file.read()

    if not raw:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    values = read_numeric_upload(raw)

    original_samples = len(values)


    # ========================================================
    # 1D CNN
    # ========================================================

    if model == "1d_cnn":

        try:

            net = get_1d_model()

        except FileNotFoundError as exc:

            raise HTTPException(
                status_code=503,
                detail=str(exc)
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=f"Could not load 1D CNN: {exc}"
            )

        x = prepare_1d_signal(
            values
        )

        tensor = torch.from_numpy(
            x
        ).view(
            1,
            1,
            TARGET_SAMPLES
        )

        with torch.inference_mode():

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

        del tensor

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
                "ECG signal analyzed using the "
                "trained Paper 1D CNN "
                "clinical-noise classifier."
            )
        }


    # ========================================================
    # RESNET-18
    # ========================================================

    elif model == "resnet18":

        try:
            net = get_resnet_model()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc)
            )
        except MemoryError:
            raise HTTPException(
                status_code=503,
                detail="Server ran out of memory attempting to load ResNet-18. Please use the 1D CNN model."
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load ResNet-18: {exc}"
            )

        try:
            # Same preprocessing as standalone
            # predict_resnet.py
            x = prepare_resnet_signal(
                values
            )

            image = ecg_to_scalogram(
                x
            )

            tensor = image.unsqueeze(
                0
            )

            with torch.inference_mode():
                logits = net(
                    tensor
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )[0]

                predicted_class = int(
                    torch.argmax(
                        probabilities
                    ).item()
                )

            del tensor, image, x
            gc.collect()

        except MemoryError:
            gc.collect()
            raise HTTPException(
                status_code=503,
                detail="Out of memory during CWT + ResNet-18 inference on this instance. Please use the lightweight 1D CNN model."
            )

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

        return {

            "success": True,

            "model": "resnet18",

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
                "samples_used": ORIGINAL_FS * WINDOW_SEC,
                "target_samples": TARGET_SAMPLES,
                "original_sampling_rate": ORIGINAL_FS,
                "target_sampling_rate": TARGET_FS,
                "window_seconds": WINDOW_SEC
            },

            "preprocessing": {
                "cwt_wavelet": "cmor1.5-1.0",
                "cwt_scales": [1, 128],
                "image_size": [224, 224]
            },

            "message": (
                "ECG signal analyzed using the "
                "trained ResNet-18 CWT "
                "clinical-noise classifier."
            )
        }


    # ========================================================
    # INVALID MODEL
    # ========================================================

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model. "
                "Choose '1d_cnn' or 'resnet18'."
            )
        )