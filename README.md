# PolarMAE

PolarMAE is a fetal ultrasound self-supervised pre-training project based on MAE and ViT. The repository includes data screening tools, valid-patch bucket generation, PolarMAE pre-training, and downstream classification, detection, and segmentation code.

## News

- 2026-04-17: Added the public project workflow, dependency file, and private-path-safe usage examples.
- 2026-04-20: Our paper "PolarMAE" is now available on [arXiv] We have also released the official project files and source code.
## Paper

- arXiv: (https://doi.org/10.48550/arXiv.2604.15893)

## Model Weights

- Baidu Netdisk: https://pan.baidu.com/s/120_fYBPgHRp-8JYcCIvOAA
- Extraction code: `r7xq`

## Workflow

The recommended pipeline is:

```text
raw ultrasound images
  -> duplicate screening
  -> valid patch counting and bucket generation
  -> PolarMAE pre-training
  -> downstream fine-tuning / evaluation
```

## Project Structure

```text
.
+-- data_tools/
|   +-- PVSS/              # Visual / semantic duplicate screening
|   +-- ABRC/              # Ultrasound sector extraction and mask generation
|   +-- compute_valid.py   # Valid patch counting and bucket generation
+-- PolarMAE-main/
    +-- PolarMAE/          # PolarMAE pre-training
    +-- Classification/    # Classification fine-tuning
    +-- Detection/         # Fetal object detection
    +-- Segmentation/      # Fetal segmentation
```

## Environment

```bash
pip install -r requirements.txt
```
For detection and segmentation experiments, install OpenMMLab packages as needed:

```bash
pip install openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmsegmentation
```

The public `requirements.txt` removes local editable installs and private filesystem paths.

## 1. Duplicate Screening

Run duplicate screening before pre-training data statistics.

Visual screening uses pHash and cosine similarity:

```bash
python data_tools/PVSS/visual_screening.py
```

Semantic screening uses MedCLIP embeddings:

```bash
python data_tools/PVSS/semantic_screening.py
```

Before running either script, edit the path constants at the top of the file, such as:

```python
RAW_IMAGE_DIR = ""
DEDUP_KEEP_DIR = ""
DEDUP_REMOVED_DIR = ""
FEATURE_CACHE_FILE = ""
```

Use the deduplicated keep directory as the input for the next step.

## 2. Valid Patch Buckets

After duplicate screening, compute the number of valid ROI patches for each image and generate bucket files.

```bash
cd PolarMAE-main/PolarMAE

python ../../data_tools/compute_valid.py \
  --root /path/to/deduplicated_train_images \
  --out ./bucket_files/valid_cnt.npy \
  --out_index ./bucket_files/valid_cnt_index.npy \
  --out_bucket ./bucket_files/valid_bucket.npy \
  --out_bucket_counts ./bucket_files/valid_bucket_counts.txt \
  --input_size 224 \
  --patch_size 16 \
```

Generated files:

```text
valid_cnt.npy
valid_cnt_index.npy
valid_bucket.npy
valid_bucket_counts.txt
```


## 3. PolarMAE Pre-training

Run pre-training after bucket generation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  main_pretrain.py \
  --selectivemae \
  --dataset_path /path/to/deduplicated_train_images \
  --bucket_path ./bucket_files/valid_bucket.npy \
  --index_path ./bucket_files/valid_cnt_index.npy \
  --output_dir ./output/pretrain_polarmae \
  --model mae_vit_base_patch16 \
  --epochs 600 \
  --batch_size 512 \
  --mask_ratio 0.75 \
  --kept_mask_ratio 0.25 \
  --lr 1.5e-3 \

```

Adjust `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, `--batch_size`, and path arguments for your own machine.
## Downstream Tasks

Classification:

```bash
cd PolarMAE-main/Classification

python main_finetune.py \
  --data_path /path/to/classification_data \
  --finetune /path/to/pretrained_checkpoint.pth \
  --output_dir ./output/classification \
```

Detection:

```bash
cd PolarMAE-main/Detection
mim train mmdet fetal/vit-b-frcn-800-proposed-fetal.py \
  --work-dir ./work_dirs/fetal_detection
```

Segmentation:

```bash
cd PolarMAE-main/Segmentation
python tools/train.py fetal/vit-b-upernet-512-proposed-ultrasound.py \
  --work-dir ./work_dirs/fetal_segmentation
```


