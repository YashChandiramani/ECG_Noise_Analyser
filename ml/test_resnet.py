import torch
import torch.nn as nn
from torchvision.models import resnet18

MODEL_PATH = "models/resnet18_noise.pt"

print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint loaded successfully!")
print("Model:", checkpoint["model_name"])
print("Task:", checkpoint["task"])
print("Classes:", checkpoint["classes"])
print("Input size:", checkpoint["input_size"])
print("Sampling rate:", checkpoint["sampling_rate"])
print("CWT wavelet:", checkpoint["cwt_wavelet"])
print("CWT scales:", checkpoint["cwt_scales"])
print("Validation accuracy:", checkpoint.get("val_accuracy"))
print("Macro F1:", checkpoint.get("macro_f1"))
print("Best epoch:", checkpoint.get("best_epoch"))

print("\nCreating ResNet-18 architecture...")

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("ResNet-18 weights loaded successfully!")
print("\nModel is ready for inference.")