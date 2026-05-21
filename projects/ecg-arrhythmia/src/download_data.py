"""
Download the PhysioNet/CinC 2017 challenge dataset into data/raw/.

Usage:
    python src/download_data.py [--data-dir data/raw]
"""

import argparse
import json
import sys
from pathlib import Path


def download(data_dir: Path) -> None:
    try:
        import wfdb
    except ImportError:
        sys.exit("wfdb not found — run: pip install wfdb>=4.1")

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PhysioNet 2017 challenge data to {data_dir} ...")
    wfdb.dl_database("challenge-2017/1.0.0", str(data_dir))

    # Verify the reference label file is present
    ref_file = data_dir / "REFERENCE.csv"
    if not ref_file.exists():
        sys.exit(
            "Download appears incomplete — REFERENCE.csv not found. "
            "Try running the script again."
        )

    _verify_and_report(data_dir, ref_file)


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
