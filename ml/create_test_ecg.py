import os
import wfdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

record_path = os.path.join(
    BASE_DIR,
    "data",
    "mitdb",
    "100"
)

print("Loading MIT-BIH record 100...")

record = wfdb.rdrecord(record_path)

ecg_signal = record.p_signal[:, 0]

print("Sampling frequency:", record.fs)
print("Total samples:", len(ecg_signal))

# Take first 5 seconds
samples = int(record.fs * 5)

test_signal = ecg_signal[:samples]

output_file = os.path.join(
    BASE_DIR,
    "test_ecg.csv"
)

pd.DataFrame({
    "ecg": test_signal
}).to_csv(
    output_file,
    index=False
)

print("\nTest ECG created successfully!")
print("Samples:", len(test_signal))
print("File:", output_file)