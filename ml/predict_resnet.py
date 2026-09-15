import os
import sys
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from scipy import signal
from torchvision.models import resnet18


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/resnet18_noise.pt"

TARGET_FS = 200
WINDOW_SEC = 5
TARGET_SAMPLES = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ImageNet normalization
MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.406]
).view(3, 1, 1)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading ResNet-18...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# RESAMPLE ECG
# ============================================================

def resample_ecg(ecg_signal, original_fs=360):

    target_samples = int(
        len(ecg_signal)
        * TARGET_FS
        / original_fs
    )

    return signal.resample(
        ecg_signal,
        target_samples
    )


# ============================================================
# CWT
# ============================================================

def ecg_to_scalogram(ecg_signal):

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

    # Add batch/channel dimensions
    scalogram = scalogram.unsqueeze(0)

    # Resize to 224 × 224
    scalogram = torch.nn.functional.interpolate(
        scalogram.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    scalogram = scalogram.squeeze()

    # Grayscale → RGB
    scalogram = scalogram.unsqueeze(0)

    scalogram = scalogram.repeat(
        3,
        1,
        1
    )

    # 0-255 → 0-1
    scalogram = scalogram / 255.0

    return scalogram


# ============================================================
# PREDICT ONE WINDOW
# ============================================================

def predict_window(model, ecg_window):

    # Resample 360 Hz → 200 Hz
    ecg_window = resample_ecg(
        ecg_window,
        original_fs=360
    )

    # Convert ECG → CWT image
    image = ecg_to_scalogram(
        ecg_window
    )

    # ImageNet normalization
    image = (
        image - MEAN
    ) / STD

    # Add batch dimension
    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    # Prediction
    with torch.no_grad():

        output = model(
            image
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return prediction, probabilities[0].cpu().numpy()


# ============================================================
# LOAD CSV
# ============================================================

def load_ecg_csv(csv_path):

    print()
    print("Loading CSV:")
    print(csv_path)

    df = pd.read_csv(
        csv_path
    )

    print("CSV shape:", df.shape)

    print("Columns:")
    print(df.columns.tolist())

    # Use first numeric column as ECG signal
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_columns) == 0:

        raise ValueError(
            "No numeric ECG column found in CSV."
        )

    ecg_column = numeric_columns[0]

    print(
        "Using ECG column:",
        ecg_column
    )

    ecg_signal = df[
        ecg_column
    ].dropna().values.astype(
        np.float32
    )

    return ecg_signal


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python predict_resnet.py <ecg.csv>"
        )

        print(
            "\nExample:"
        )

        print(
            "python predict_resnet.py test_ecg.csv"
        )

        return

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):

        print(
            "ERROR: File not found:",
            csv_path
        )

        return

    model = load_model()

    ecg_signal = load_ecg_csv(
        csv_path
    )

    required_samples = (
        WINDOW_SEC * 360
    )

    if len(ecg_signal) < required_samples:

        raise ValueError(
            f"ECG must contain at least "
            f"{required_samples} samples."
        )

    # Use first 5-second window
    ecg_window = ecg_signal[
        :required_samples
    ]

    print()
    print("Processing 5-second ECG window...")

    prediction, probabilities = predict_window(
        model,
        ecg_window
    )

    class_names = {
        0: "Clean",
        1: "Noisy"
    }

    print()
    print("=" * 50)
    print("RESNET-18 PREDICTION")
    print("=" * 50)

    print(
        "Prediction:",
        class_names[prediction]
    )

    print(
        f"Clean probability: "
        f"{probabilities[0]:.4f}"
    )

    print(
        f"Noisy probability: "
        f"{probabilities[1]:.4f}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()