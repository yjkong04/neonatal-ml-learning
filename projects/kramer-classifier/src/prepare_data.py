"""
Prepare HAM10000 dataset into ImageFolder format.

Input:  flat folder of .jpg images + HAM10000_metadata CSV
Output: data/train/{class}/, data/val/{class}/, data/test/{class}/

Run from the project root:
    python3 src/prepare_data.py \
        --images_dir ~/Downloads/dataverse_files/images \
        --metadata   ~/Downloads/dataverse_files/HAM10000_metadata \
        --output_dir data
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main(images_dir: str, metadata_path: str, output_dir: str):
    images_dir = Path(images_dir).expanduser()
    metadata_path = Path(metadata_path).expanduser()
    output_dir = Path(output_dir)

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df)} rows")
    print(df['dx'].value_counts().to_string())

    # Drop duplicate lesion entries (same lesion, different image angle).
    # Keep one image per lesion so the same lesion never appears in both train and test.
    df = df.drop_duplicates(subset='lesion_id')
    print(f"\nAfter deduplication: {len(df)} unique lesions")

    # Stratified split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['dx'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['dx'], random_state=42)

    print(f"\nSplit sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    copied, missing = 0, 0
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            img_src = images_dir / f"{row['image_id']}.jpg"
            if not img_src.exists():
                missing += 1
                continue
            dest_dir = output_dir / split_name / row['dx']
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_src, dest_dir / img_src.name)
            copied += 1

    print(f"\nDone. Copied: {copied}, Missing: {missing}")
    if missing > 0:
        print("Missing images — check that both part_1 and part_2 zips are extracted into --images_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True, help='Folder containing all .jpg images')
    parser.add_argument('--metadata',   required=True, help='Path to HAM10000_metadata CSV')
    parser.add_argument('--output_dir', default='data', help='Where to write train/val/test folders')
    args = parser.parse_args()
    main(args.images_dir, args.metadata, args.output_dir)
