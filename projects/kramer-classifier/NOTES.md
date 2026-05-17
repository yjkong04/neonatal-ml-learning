# Kramer Classifier — Notes and Data Setup

## Dataset

**HAM10000** (Harvard Dataverse) — transfer learning on dermoscopy imagery as a proxy for neonatal jaundice photos.
Same technique as the production neonatal jaundice classifier (pretrained backbone, fine-tuned classifier).
The training pipeline is identical; the dataset is adult skin lesions, not neonatal jaundice.

## Data setup (run once on a fresh machine)

**1. Download HAM10000** from Harvard Dataverse (free, no login):
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Download `dataverse_files.zip` (~3GB) and unzip it.

**2. Extract the images** into a flat folder:
```bash
cd ~/Downloads/dataverse_files
unzip HAM10000_images_part_1.zip -d images
unzip HAM10000_images_part_2.zip -d images
```

**3. Install dependencies:**
```bash
pip3 install torch torchvision fastai Pillow matplotlib scikit-learn pandas
```

**4. Split into train/val/test** (run from project root):
```bash
python3 src/prepare_data.py \
  --images_dir ~/Downloads/dataverse_files/images \
  --metadata   ~/Downloads/dataverse_files/HAM10000_metadata \
  --output_dir data
```

Expected output: 5229 train / 1120 val / 1121 test images across 7 classes.
The `data/` folder is gitignored — never commit images.
