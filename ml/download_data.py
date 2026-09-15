import os
import wfdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MITDB_DIR = os.path.join(BASE_DIR, "data", "mitdb")
NSTDB_DIR = os.path.join(BASE_DIR, "data", "nstdb")

os.makedirs(MITDB_DIR, exist_ok=True)
os.makedirs(NSTDB_DIR, exist_ok=True)

MITDB_RECORDS = [
    "100",
    "101",
    "103",
    "105",
    "112",
    "115",
    "121",
    "200",
]

NOISE_RECORDS = [
    "em",
    "ma",
]


def download_mitdb():
    print("\nDownloading MIT-BIH records...")

    for record in MITDB_RECORDS:
        print(f"Downloading record {record}...")

        wfdb.dl_database(
            "mitdb",
            dl_dir=MITDB_DIR,
            records=[record]
        )

    print("MIT-BIH download complete.")


def download_noise():
    print("\nDownloading noise records...")

    wfdb.dl_database(
        "nstdb",
        dl_dir=NSTDB_DIR,
        records=NOISE_RECORDS
    )

    print("Noise database download complete.")


if __name__ == "__main__":
    print("ECG dataset downloader")
    print("======================")

    download_mitdb()
    download_noise()

    print("\nAll required ECG data downloaded successfully!")