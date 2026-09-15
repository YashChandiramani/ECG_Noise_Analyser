import os
import random
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as sign
import pywt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_FS = 200
WINDOW_SEC = 5
TARGET_SAMPLES = 1000

CACHE_DIR = "data/cwt_cache"
MODEL_DIR = "models"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_noise.pt"
)

print("=" * 60)
print("ECG CLINICAL NOISE CLASSIFICATION")
print("ResNet-18 + CWT")
print("=" * 60)

print()
print("Using device:", DEVICE)


# ============================================================
# RESAMPLING
# ============================================================

def resample_segment(sig, orig_fs=360, target_fs=200):

    num_samples = int(len(sig) * target_fs / orig_fs)

    return sign.resample(sig, num_samples)


# ============================================================
# NOISE INJECTION
# ============================================================

def inject_noise_at_snr(clean_seg, noise_seg, target_snr_db):

    p_clean = np.mean(clean_seg ** 2)
    p_noise = np.mean(noise_seg ** 2)

    if p_noise == 0:
        return clean_seg

    target_p_noise = p_clean / (
        10 ** (target_snr_db / 10.0)
    )

    scaling_factor = np.sqrt(
        target_p_noise / p_noise
    )

    return clean_seg + scaling_factor * noise_seg


# ============================================================
# DATASET SYNTHESIS
# ============================================================

def synthesize_paper_dataset():

    print()
    print("Loading noise records...")

    em_record = wfdb.rdrecord(
        "em",
        pn_dir="nstdb"
    )

    ma_record = wfdb.rdrecord(
        "ma",
        pn_dir="nstdb"
    )

    em_noise_raw = em_record.p_signal[:, 0]
    ma_noise_raw = ma_record.p_signal[:, 0]

    clean_record_ids = [
        "100",
        "101",
        "103",
        "105",
        "112",
        "115",
        "121",
        "200"
    ]

    fs_orig = 360
    samples_per_window_orig = WINDOW_SEC * fs_orig

    X_data = []
    y_labels = []
    patient_ids = []

    for rec_id in clean_record_ids:

        print(f"Processing record {rec_id}...")

        record = wfdb.rdrecord(
            rec_id,
            pn_dir="mitdb"
        )

        clean_signal = record.p_signal[:, 0]

        num_windows = (
            len(clean_signal)
            // samples_per_window_orig
        )

        for i in range(num_windows):

            start = i * samples_per_window_orig
            end = start + samples_per_window_orig

            clean_chunk = clean_signal[start:end]

            noise_start = random.randint(
                0,
                len(em_noise_raw)
                - samples_per_window_orig
                - 1
            )

            noise_end = (
                noise_start
                + samples_per_window_orig
            )

            if random.random() > 0.5:
                noise_chunk = em_noise_raw[
                    noise_start:noise_end
                ]
            else:
                noise_chunk = ma_noise_raw[
                    noise_start:noise_end
                ]

            # ------------------------------------------------
            # Four conditions
            #
            # None -> clean
            # 18 dB -> clean
            # 6 dB -> noisy
            # -6 dB -> noisy
            # ------------------------------------------------

            for snr, label in [
                (None, 0),
                (18, 0),
                (6, 1),
                (-6, 1)
            ]:

                if snr is None:

                    segment = clean_chunk

                else:

                    segment = inject_noise_at_snr(
                        clean_chunk,
                        noise_chunk,
                        snr
                    )

                segment = resample_segment(
                    segment,
                    fs_orig,
                    TARGET_FS
                )

                X_data.append(segment)
                y_labels.append(label)
                patient_ids.append(rec_id)

    X_data = np.asarray(
        X_data,
        dtype=np.float32
    )

    y_labels = np.asarray(
        y_labels,
        dtype=np.int64
    )

    patient_ids = np.asarray(
        patient_ids
    )

    return X_data, y_labels, patient_ids


# ============================================================
# CWT CONVERSION
# ============================================================

def ecg_to_scalogram(signal):

    # Morlet wavelet used in the project
    scales = np.arange(1, 129)

    coefficients, frequencies = pywt.cwt(
        signal,
        scales,
        "cmor1.5-1.0"
    )

    # Magnitude
    scalogram = np.abs(coefficients)

    # Log compression
    scalogram = np.log1p(scalogram)

    # Normalize each image
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

    # Convert to uint8
    scalogram = (
        scalogram * 255
    ).astype(np.uint8)

    # Resize to 224 x 224
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

    # Convert grayscale -> RGB
    scalogram = scalogram.unsqueeze(0)

    scalogram = scalogram.repeat(
        3,
        1,
        1
    )

    # Convert 0-255 -> 0-1
    scalogram = scalogram / 255.0

    return scalogram


# ============================================================
# PRECOMPUTE CWT CACHE
# ============================================================

def create_cwt_cache(X, y, patients):

    print()
    print("=" * 60)
    print("CREATING CWT CACHE")
    print("=" * 60)

    cache_dir = os.path.join(
        CACHE_DIR,
        "images"
    )

    os.makedirs(
        cache_dir,
        exist_ok=True
    )

    metadata_file = os.path.join(
        CACHE_DIR,
        "metadata.pt"
    )

    # --------------------------------------------------------
    # Check whether cache already exists
    # --------------------------------------------------------

    expected_count = len(X)

    existing_files = [
        f for f in os.listdir(cache_dir)
        if f.endswith(".pt")
    ]

    if (
        len(existing_files) == expected_count
        and os.path.exists(metadata_file)
    ):

        print()
        print("CWT cache already exists.")

        print(
            "Cached images:",
            len(existing_files)
        )

        metadata = torch.load(
            metadata_file,
            map_location="cpu",
            weights_only = False
        )

        return (
            cache_dir,
            metadata["labels"],
            metadata["patients"]
        )

    # --------------------------------------------------------
    # If an incomplete cache exists, remove it
    # --------------------------------------------------------

    if len(existing_files) > 0:

        print()
        print(
            "Incomplete CWT cache detected."
        )

        print(
            "Removing old cache..."
        )

        for filename in existing_files:

            os.remove(
                os.path.join(
                    cache_dir,
                    filename
                )
            )

    if os.path.exists(metadata_file):

        os.remove(
            metadata_file
        )

    # --------------------------------------------------------
    # Generate CWT images
    # --------------------------------------------------------

    print()
    print(
        f"Generating {len(X)} scalograms..."
    )

    for i, signal in enumerate(X):

        image_file = os.path.join(
            cache_dir,
            f"{i:05d}.pt"
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if i % 250 == 0:

            print(
                f"CWT progress: "
                f"{i}/{len(X)}"
            )

        # ----------------------------------------------------
        # Convert ECG → CWT scalogram
        # ----------------------------------------------------

        image = ecg_to_scalogram(
            signal
        )

        # ----------------------------------------------------
        # Save immediately to disk
        # ----------------------------------------------------

        torch.save(
            image,
            image_file
        )

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del image

    # --------------------------------------------------------
    # Save labels and patient IDs separately
    # --------------------------------------------------------

    labels = torch.tensor(
        y,
        dtype=torch.long
    )

    patients = np.asarray(
        patients
    )

    torch.save(
        {
            "labels": labels,
            "patients": patients
        },
        metadata_file
    )

    print()
    print(
        "CWT cache saved successfully."
    )

    print(
        "Cache directory:",
        cache_dir
    )

    print(
        "Number of images:",
        len(X)
    )

    return (
        cache_dir,
        labels,
        patients
    )


# ============================================================
# PYTORCH DATASET
# ============================================================

class CWTDataset(Dataset):

    def __init__(
        self,
        image_dir,
        indices,
        labels,
        mean,
        std
    ):

        self.image_dir = image_dir
        self.indices = np.asarray(
            indices
        )

        self.labels = labels

        self.mean = mean
        self.std = std

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):

        real_index = int(
            self.indices[index]
        )

        image_file = os.path.join(
            self.image_dir,
            f"{real_index:05d}.pt"
        )

        image = torch.load(
            image_file,
            map_location="cpu",
            weights_only = True
        )

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        image = (
            image - self.mean
        ) / self.std

        label = self.labels[
            real_index
        ]

        return (
            image,
            label
        )


# ============================================================
# RESNET-18
# ============================================================

def create_resnet18():

    print()
    print("Loading pretrained ResNet-18...")

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )

    # Replace ImageNet classifier
    # with binary ECG classifier

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=1
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    best_f1 = -1

    best_state = None

    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    DEVICE
                )

                outputs = model(
                    images
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1
                    )
                    .cpu()
                    .numpy()
                )

                all_predictions.extend(
                    predictions
                )

                all_labels.extend(
                    labels.numpy()
                )

        val_accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        val_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro"
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1

            best_state = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
    }

            torch.save(
        {
            "model_state_dict": best_state,
            "model_name": "ResNet-18",
            "task": "Clinical ECG Noise Classification",
            "classes": {
                0: "Clean",
                1: "Noisy"
            },
            "input_size": [3, 224, 224],
            "sampling_rate": TARGET_FS,
            "cwt_wavelet": "cmor1.5-1.0",
            "cwt_scales": [1, 128],
            "val_accuracy": val_accuracy,
            "macro_f1": val_f1,
            "best_epoch": epoch + 1
        },
        MODEL_PATH
    )

        print("✓ New best model saved to:", os.path.abspath(MODEL_PATH))

        scheduler.step()

    # Restore best model
    model.load_state_dict(
        best_state
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1: Create ECG dataset
    # --------------------------------------------------------

    X_all, y_all, pid_all = (
        synthesize_paper_dataset()
    )

    print()
    print("Dataset created.")

    print(
        "X shape:",
        X_all.shape
    )

    print(
        "y shape:",
        y_all.shape
    )

    print(
        "Unique labels:",
        np.unique(y_all)
    )

    print(
        "Patients:",
        np.unique(pid_all)
    )

    # --------------------------------------------------------
    # STEP 2: Patient-isolated split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            X_all,
            y_all,
            groups=pid_all
        )
    )

    print()

    print(
        "Train shape:",
        X_all[train_idx].shape
    )

    print(
        "Validation shape:",
        X_all[val_idx].shape
    )

    print(
        "Training patients:",
        np.unique(pid_all[train_idx])
    )

    print(
        "Validation patients:",
        np.unique(pid_all[val_idx])
    )

    # --------------------------------------------------------
    # STEP 3: CWT cache
    # --------------------------------------------------------

    cwt_dir, labels, patients = (
        create_cwt_cache(
            X_all,
            y_all,
            pid_all
        )
    )

    # --------------------------------------------------------
    # STEP 5: ImageNet normalization
    # --------------------------------------------------------

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.406]
    ).view(3, 1, 1)

    # --------------------------------------------------------
    # STEP 6: PyTorch datasets
    # --------------------------------------------------------

    train_dataset = CWTDataset(
        image_dir=cwt_dir,
        indices=train_idx,
        labels=labels,
        mean=mean,
        std=std
    )

    val_dataset = CWTDataset(
        image_dir=cwt_dir,
        indices=val_idx,
        labels=labels,
        mean=mean,
        std=std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print()
    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # --------------------------------------------------------
    # STEP 7: ResNet
    # --------------------------------------------------------

    model = create_resnet18()

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # STEP 8: Training
    # --------------------------------------------------------

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=15
    )

    # --------------------------------------------------------
    # STEP 9: Final evaluation
    # --------------------------------------------------------

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            actual.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        actual,
        predictions
    )

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro"
    )

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    print()
    print("Classification Report:")

    print(
        classification_report(
            actual,
            predictions,
            target_names=[
                "Clean",
                "Noisy"
            ]
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            actual,
            predictions
        )
    )

    # --------------------------------------------------------
    # STEP 10: Save model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "model_name":
                "ResNet-18",

            "task":
                "Clinical ECG Noise Classification",

            "classes":
                {
                    0: "Clean",
                    1: "Noisy"
                },

            "input_size":
                [3, 224, 224],

            "sampling_rate":
                TARGET_FS,

            "cwt_wavelet":
                "cmor1.5-1.0",

            "cwt_scales":
                [1, 128],

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1
        },
        MODEL_PATH
    )

    print()
    print(
        "Model saved to:"
    )

    print(
        os.path.abspath(
            MODEL_PATH
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()