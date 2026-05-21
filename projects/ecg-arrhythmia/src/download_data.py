"""
Download the PhysioNet/CinC 2017 challenge dataset into data/raw/.

Usage:
    python src/download_data.py [--data-dir data/raw]
"""

import argparse
import json
import urllib.request
from pathlib import Path

BASE_URL = "https://physionet.org/files/challenge-2017/1.0.0/training"


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file list from PhysioNet ...")
    record_names = _fetch_record_list()
    print(f"Found {len(record_names)} records. Downloading to {data_dir} ...")

    # Each record has a .hea header and a .mat signal file
    extensions = [".hea", ".mat"]
    total_files = len(record_names) * len(extensions) + 1  # +1 for REFERENCE.csv
    downloaded = 0

    for name in record_names:
        for ext in extensions:
            _fetch_file(f"{BASE_URL}/{name}{ext}", data_dir / f"{name}{ext}")
            downloaded += 1
            if downloaded % 500 == 0:
                print(f"  {downloaded}/{total_files} files ...")

    ref_dest = data_dir / "REFERENCE.csv"
    _fetch_file(
        "https://physionet.org/files/challenge-2017/1.0.0/REFERENCE.csv",
        ref_dest,
    )

    _verify_and_report(data_dir, ref_dest)


def _fetch_record_list() -> list:
    """Return record names by reading the WFDB index file (RECORDS)."""
    records_url = f"{BASE_URL}/RECORDS"
    with urllib.request.urlopen(records_url) as resp:
        return [line.strip() for line in resp.read().decode().splitlines() if line.strip()]


def _fetch_file(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _verify_and_report(data_dir: Path, ref_file: Path) -> None:
    import csv
    from collections import Counter

    counts: Counter = Counter()
    records: list[str] = []

    with open(ref_file) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            record_name, label = row[0].strip(), row[1].strip()
            counts[label] += 1
            records.append(record_name)

    total = sum(counts.values())
    print(f"\nDownload complete — {total} records in {data_dir}")
    print("Class distribution:")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:10s} {n:5d}  ({100 * n / total:.1f}%)")

    # Save a small manifest so other scripts can locate data without re-parsing REFERENCE.csv
    manifest = {
        "data_dir": str(data_dir),
        "total_records": total,
        "class_counts": dict(counts),
        "records": records,
    }
    manifest_path = data_dir.parent / "processed" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PhysioNet 2017 ECG data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
        help="Destination directory (default: data/raw)",
    )
    args = parser.parse_args()
    download(args.data_dir)
