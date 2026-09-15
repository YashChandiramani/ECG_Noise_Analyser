import os
import random
import numpy as np
import wfdb
import scipy.signal as sign

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MITDB_DIR = os.path.join(BASE_DIR, "data", "mitdb")
NSTDB_DIR = os.path.join(BASE_DIR, "data", "nstdb")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. MODEL
# ============================================================

class Paper1DCNN(nn.Module):

    def __init__(self, num_classes=2):

        super(Paper1DCNN, self).__init__()

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

        x = self.fc_relu1(self.fc1(x))
        x = self.fc_relu2(self.fc2(x))

        return self.classifier(x)


# ============================================================
# 4. RESAMPLING
# ============================================================

def resample_segment(
    sig,
    orig_fs=360,
    target_fs=200
):

    num_samples = int(
        len(sig) * target_fs / orig_fs
    )

    return sign.resample(
        sig,
        num_samples
    )


# ============================================================
# 5. NOISE INJECTION
# ============================================================

def inject_noise_at_snr(
    clean_seg,
    noise_seg,
    target_snr_db
):

    p_clean = np.mean(
        clean_seg ** 2
    )

    p_noise = np.mean(
        noise_seg ** 2
    )

    if p_noise == 0:
        return clean_seg

    target_p_noise = (
        p_clean /
        (10 ** (target_snr_db / 10.0))
    )

    scaling_factor = np.sqrt(
        target_p_noise / p_noise
    )

    return clean_seg + (
        scaling_factor * noise_seg
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

def load_record(record_id, database_dir):

    record_path = os.path.join(
        database_dir,
        record_id
    )

    record = wfdb.rdrecord(
        record_path
    )

    return record.p_signal[:, 0]


# ============================================================
# 7. CREATE CLINICAL-NOISE DATASET
# ============================================================

def synthesize_paper_dataset():

    print("\nLoading noise records...")

    em_noise_raw = load_record(
        "em",
        NSTDB_DIR
    )

    ma_noise_raw = load_record(
        "ma",
        NSTDB_DIR
    )

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
    fs_target = 200
    window_sec = 5

    samples_per_window_orig = (
        window_sec * fs_orig
    )

    X_data = []
    y_labels = []
    patient_ids = []

    print("\nCreating synthetic ECG dataset...")

    for rec_id in clean_record_ids:

        print(
            f"Processing record {rec_id}..."
        )

        clean_signal = load_record(
            rec_id,
            MITDB_DIR
        )

        num_windows = (
            len(clean_signal)
            // samples_per_window_orig
        )

        print(
            f"  Windows: {num_windows}"
        )

        for i in range(num_windows):

            start = (
                i *
                samples_per_window_orig
            )

            end = (
                start +
                samples_per_window_orig
            )

            clean_chunk = (
                clean_signal[start:end]
            )

            # Make sure enough noise exists
            max_noise_start = (
                len(em_noise_raw)
                - samples_per_window_orig
                - 1
            )

            noise_start = random.randint(
                0,
                max_noise_start
            )

            noise_end = (
                noise_start +
                samples_per_window_orig
            )

            if random.random() > 0.5:

                noise_chunk = (
                    em_noise_raw[
                        noise_start:noise_end
                    ]
                )

            else:

                noise_chunk = (
                    ma_noise_raw[
                        noise_start:noise_end
                    ]
                )

            # Four conditions from the notebook
            conditions = [
                (None, 0),
                (18, 0),
                (6, 1),
                (-6, 1)
            ]

            for snr, label in conditions:

                if snr is None:

                    seg = clean_chunk.copy()

                else:

                    seg = inject_noise_at_snr(
                        clean_chunk,
                        noise_chunk,
                        snr
                    )

                # 360 Hz -> 200 Hz
                seg = resample_segment(
                    seg,
                    fs_orig,
                    fs_target
                )

                X_data.append(seg)
                y_labels.append(label)
                patient_ids.append(rec_id)

    X_data = np.array(
        X_data,
        dtype=np.float32
    )

    y_labels = np.array(
        y_labels,
        dtype=np.int64
    )

    patient_ids = np.array(
        patient_ids
    )

    # Add channel dimension
    # (N, 1000) -> (N, 1, 1000)

    X_data = np.expand_dims(
        X_data,
        axis=1
    )

    return (
        X_data,
        y_labels,
        patient_ids
    )


# ============================================================
# 8. TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.99,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_f1 = -1

    best_state = None

    patience = 7
    patience_counter = 0

    for epoch in range(epochs):

        # -------------------------
        # Training
        # -------------------------

        model.train()

        train_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * X_batch.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        val_predictions = []
        val_targets = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)

                outputs = model(
                    X_batch
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1
                    )
                    .cpu()
                    .numpy()
                )

                val_predictions.extend(
                    predictions
                )

                val_targets.extend(
                    y_batch.numpy()
                )

        val_f1 = f1_score(
            val_targets,
            val_predictions,
            average="macro"
        )

        val_accuracy = accuracy_score(
            val_targets,
            val_predictions
        )

        scheduler.step(val_f1)

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

            print(
                "  ✓ New best model"
            )

        else:

            patience_counter += 1

        # -------------------------
        # Early stopping
        # -------------------------

        if patience_counter >= patience:

            print(
                "\nEarly stopping."
            )

            break

    # Restore best model

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "ECG Clinical Noise Classifier"
    )

    print(
        "Paper 1D CNN"
    )

    print(
        "======================================"
    )

    # ----------------------------------
    # Generate dataset
    # ----------------------------------

    X_all, y_all, patient_ids = (
        synthesize_paper_dataset()
    )

    print(
        "\nDataset created."
    )

    print(
        f"X shape: {X_all.shape}"
    )

    print(
        f"y shape: {y_all.shape}"
    )

    print(
        f"Unique labels: {np.unique(y_all)}"
    )

    print(
        f"Patients: {np.unique(patient_ids)}"
    )

    # ----------------------------------
    # Patient-isolated split
    # ----------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SEED
    )

    train_idx, val_idx = next(
        splitter.split(
            X_all,
            y_all,
            groups=patient_ids
        )
    )

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    print(
        "\nTrain shape:",
        X_train.shape
    )

    print(
        "Validation shape:",
        X_val.shape
    )

    print(
        "\nTraining patients:",
        np.unique(
            patient_ids[train_idx]
        )
    )

    print(
        "Validation patients:",
        np.unique(
            patient_ids[val_idx]
        )
    )

    # ----------------------------------
    # Convert to PyTorch tensors
    # ----------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long
    )

    # ----------------------------------
    # DataLoaders
    # ----------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    # ----------------------------------
    # Model
    # ----------------------------------

    model = Paper1DCNN(
        num_classes=2
    )

    print(
        "\nModel created."
    )

    # ----------------------------------
    # Train
    # ----------------------------------

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=50
    )

    # ----------------------------------
    # Final evaluation
    # ----------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    model.eval()

    predictions = []

    targets = []

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)

            outputs = model(
                X_batch
            )

            preds = torch.argmax(
                outputs,
                dim=1
            ).cpu().numpy()

            predictions.extend(
                preds
            )

            targets.extend(
                y_batch.numpy()
            )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    print(
        "\n======================================"
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "======================================"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            targets,
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
            targets,
            predictions
        )
    )

    # ----------------------------------
    # Save model
    # ----------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "paper_1d_cnn_noise.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        f"\n✓ Model saved to:"
    )

    print(
        model_path
    )

    print(
        "\nTraining complete!"
    )


if __name__ == "__main__":

    main()