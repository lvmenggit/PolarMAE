import os
import argparse
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import hashlib

import torch
from torchvision import transforms

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


DEFAULT_ROOT = ""
DEFAULT_INPUT_SIZE = 224
DEFAULT_PATCH_SIZE = 16
DEFAULT_ROI_TAU = 0.1
DEFAULT_MASK_SUFFIX = "_mask"
DEFAULT_MASK_EXT = ".png"
DEFAULT_OUT = "./valid_cnt.npy"
DEFAULT_OUT_INDEX = "./valid_cnt_index.npy"
DEFAULT_OUT_BUCKET = "./valid_bucket.npy"
DEFAULT_OUT_BUCKET_COUNTS = "./valid_bucket_counts.txt"
DEFAULT_BIN_SIZE = 16
DEFAULT_PRINT_EVERY = 2000
DEFAULT_AUG_MODE = "train"  
DEFAULT_NUM_REPEATS = 20
DEFAULT_CROP_SCALE_LOW = 0.2
DEFAULT_CROP_SCALE_HIGH = 1.0
DEFAULT_HFLIP_PROB = 0.5
DEFAULT_SEED = 1
DEFAULT_AGG = "q20" 


def _stable_u32_from_str(s: str) -> int:

    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], byteorder="little", signed=False)


def _augment_mask_like_train(
    mask_img: Image.Image,
    input_size: int,
    crop_scale: tuple,
    hflip_prob: float,
    seed: int | None,
) -> Image.Image:
    if seed is not None:
        torch.manual_seed(int(seed))

    i, j, h, w = transforms.RandomResizedCrop.get_params(
        mask_img, scale=crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
    )
    mask_aug = TF.resized_crop(
        mask_img,
        i,
        j,
        h,
        w,
        (int(input_size), int(input_size)),
        interpolation=InterpolationMode.NEAREST,
    )
    if float(hflip_prob) > 0.0 and torch.rand(1).item() < float(hflip_prob):
        mask_aug = TF.hflip(mask_aug)
    return mask_aug


def list_image_files(root: str, mask_suffix: str):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            lower = fn.lower()
            if not lower.endswith(IMG_EXTS):
                continue
            stem, _ = os.path.splitext(fn)
            if stem.endswith(mask_suffix):
                continue
            files.append(os.path.join(dirpath, fn))
    return sorted(files)


def mask_path_from_img(img_path: str, mask_suffix: str, mask_ext: str):
    base, _ = os.path.splitext(img_path)
    return base + f"{mask_suffix}{mask_ext}"


def _compute_valid_cnt_from_mask_pil(mask_pil: Image.Image, patch_size: int, roi_tau: float) -> int:
    """mask_pil 已经是 [input_size,input_size] 的 mask（可为增强后）。"""
    m = np.asarray(mask_pil)
    if m.ndim == 3:

        m = m[..., 0]
    m = (m > 0).astype(np.float32)

    p = int(patch_size)
    H, W = int(m.shape[0]), int(m.shape[1])
    Gh, Gw = H // p, W // p
    if Gh <= 0 or Gw <= 0:
        return 0
    m = m[: Gh * p, : Gw * p]
    pooled = m.reshape(Gh, p, Gw, p).mean(axis=(1, 3))  # [Gh,Gw]

    patch_valid = pooled.reshape(-1) >= float(roi_tau)
    vi = int(patch_valid.sum())

    if vi <= 0:
        vi = int(Gh * Gw)
    return vi


def compute_valid_cnt_one(mask_img: Image.Image, input_size: int, patch_size: int, roi_tau: float) -> int:
    mask_rs = TF.resize(mask_img, [int(input_size), int(input_size)], interpolation=InterpolationMode.NEAREST)
    return _compute_valid_cnt_from_mask_pil(mask_rs, patch_size, roi_tau)


def compute_valid_cnt_one_path(
    img_path: str,
    input_size: int,
    patch_size: int,
    roi_tau: float,
    mask_suffix: str,
    mask_ext: str,
) -> float:
    mpath = mask_path_from_img(img_path, mask_suffix, mask_ext)
    Gh = int(input_size) // int(patch_size)
    full = float(Gh * Gh)
    if not os.path.exists(mpath):
        return full

    mask = Image.open(mpath).convert("L")
    return float(compute_valid_cnt_one(mask, input_size, patch_size, roi_tau))


def compute_valid_cnt_one_path_aug(
    img_path: str,
    input_size: int,
    patch_size: int,
    roi_tau: float,
    mask_suffix: str,
    mask_ext: str,
    num_repeats: int,
    crop_scale_low: float,
    crop_scale_high: float,
    hflip_prob: float,
    agg: str,
    seed: int,
) -> float:
    mpath = mask_path_from_img(img_path, mask_suffix, mask_ext)
    Gh = int(input_size) // int(patch_size)
    full = float(Gh * Gh)
    if not os.path.exists(mpath):
        return full

    mask = Image.open(mpath).convert("L")
    r = int(num_repeats)
    if r <= 1:
        # 仍然走一次“训练同款增强”
        sample_seed = None
        if int(seed) >= 0:
            sample_seed = int(seed) ^ int(_stable_u32_from_str(img_path))
        mask_aug = _augment_mask_like_train(
            mask,
            input_size=input_size,
            crop_scale=(float(crop_scale_low), float(crop_scale_high)),
            hflip_prob=hflip_prob,
            seed=sample_seed,
        )
        return float(_compute_valid_cnt_from_mask_pil(mask_aug, patch_size, roi_tau))

    vals: list[float] = []
    base = None
    if int(seed) >= 0:
        base = (int(seed) * 1000003) ^ int(_stable_u32_from_str(img_path))

    for k in range(r):
        sample_seed = None
        if base is not None:
            sample_seed = base + k * 9176

        mask_aug = _augment_mask_like_train(
            mask,
            input_size=input_size,
            crop_scale=(float(crop_scale_low), float(crop_scale_high)),
            hflip_prob=hflip_prob,
            seed=sample_seed,
        )
        vals.append(float(_compute_valid_cnt_from_mask_pil(mask_aug, patch_size, roi_tau)))

    if not vals:
        return full

    agg = str(agg).lower()
    if agg == "mean":
        return float(np.mean(vals))
    if agg == "q10":
        return float(np.quantile(vals, 0.10))
    if agg == "q20":
        return float(np.quantile(vals, 0.20))
    # fallback
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT, help="数据根目录（递归扫描图片）")
    ap.add_argument("--input_size", type=int, default=DEFAULT_INPUT_SIZE)
    ap.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    ap.add_argument("--roi_tau", type=float, default=DEFAULT_ROI_TAU)

    ap.add_argument("--mask_suffix", type=str, default=DEFAULT_MASK_SUFFIX)
    ap.add_argument("--mask_ext", type=str, default=DEFAULT_MASK_EXT)

    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--out_index", type=str, default=DEFAULT_OUT_INDEX,
                    help="保存与 valid_cnt 对齐的图片路径（np.array of str）")
    ap.add_argument("--out_bucket", type=str, default=DEFAULT_OUT_BUCKET,
                    help="保存与 valid_cnt 对齐的 bucket（np.array of int）")
    ap.add_argument("--out_bucket_counts", type=str, default=DEFAULT_OUT_BUCKET_COUNTS,
                    help="保存每个桶的样本数（文本日志）")
    ap.add_argument("--bin_size", type=int, default=DEFAULT_BIN_SIZE,
                    help="分桶区间宽度（例如 8 表示每 8 个 patch 为一桶）")
    ap.add_argument("--print_every", type=int, default=DEFAULT_PRINT_EVERY)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="进程数（>0 启用多进程；0 为单进程）")
    ap.add_argument("--aug_mode", type=str, default=DEFAULT_AUG_MODE, choices=["train", "resize"],
                    help="train: 与训练一致的随机裁剪+翻转；resize: 仅 resize（确定性）")
    ap.add_argument("--num_repeats", type=int, default=DEFAULT_NUM_REPEATS,
                    help="每张图随机增强采样次数（仅 aug_mode=train 生效）")
    ap.add_argument("--crop_scale_low", type=float, default=DEFAULT_CROP_SCALE_LOW,
                    help="RandomResizedCrop scale 下界（与训练一致）")
    ap.add_argument("--crop_scale_high", type=float, default=DEFAULT_CROP_SCALE_HIGH,
                    help="RandomResizedCrop scale 上界（与训练一致）")
    ap.add_argument("--hflip", type=float, default=DEFAULT_HFLIP_PROB,
                    help="水平翻转概率（与训练一致）")
    ap.add_argument("--agg", type=str, default=DEFAULT_AGG, choices=["mean", "q10", "q20"],
                    help="多次随机增强后的聚合口径：mean 或分位数 q10/q20（更保守，训练时更不易低于阈值）")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=">=0 则每个样本增强可复现；<0 则完全随机")
    args = ap.parse_args()

    img_files = list_image_files(args.root, args.mask_suffix)
    if len(img_files) == 0:
        raise RuntimeError(f"No images found under: {args.root}")

    # aug_mode=train 时输出为“平均有效 patch 数”（float）；resize 时仍是 float，但为整数值
    valid_cnt = np.zeros((len(img_files),), dtype=np.float32)

    if args.num_workers and int(args.num_workers) > 0:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as ex:
            if args.aug_mode == "train":
                fn = compute_valid_cnt_one_path_aug
                it = ex.map(
                    fn,
                    img_files,
                    [args.input_size] * len(img_files),
                    [args.patch_size] * len(img_files),
                    [args.roi_tau] * len(img_files),
                    [args.mask_suffix] * len(img_files),
                    [args.mask_ext] * len(img_files),
                    [args.num_repeats] * len(img_files),
                    [args.crop_scale_low] * len(img_files),
                    [args.crop_scale_high] * len(img_files),
                    [args.hflip] * len(img_files),
                    [args.agg] * len(img_files),
                    [args.seed] * len(img_files),
                    chunksize=32,
                )
            else:
                fn = compute_valid_cnt_one_path
                it = ex.map(
                    fn,
                    img_files,
                    [args.input_size] * len(img_files),
                    [args.patch_size] * len(img_files),
                    [args.roi_tau] * len(img_files),
                    [args.mask_suffix] * len(img_files),
                    [args.mask_ext] * len(img_files),
                    chunksize=64,
                )
            for i, vi in enumerate(it):
                valid_cnt[i] = float(vi)
                if args.print_every > 0 and (i + 1) % args.print_every == 0:
                    print(f"[{i+1}/{len(img_files)}] vi={vi}  path={img_files[i]}")
    else:
        for i, img_path in enumerate(img_files):
            if args.aug_mode == "train":
                vi = compute_valid_cnt_one_path_aug(
                    img_path,
                    args.input_size,
                    args.patch_size,
                    args.roi_tau,
                    args.mask_suffix,
                    args.mask_ext,
                    args.num_repeats,
                    args.crop_scale_low,
                    args.crop_scale_high,
                    args.hflip,
                    args.agg,
                    args.seed,
                )
            else:
                vi = compute_valid_cnt_one_path(
                    img_path,
                    args.input_size,
                    args.patch_size,
                    args.roi_tau,
                    args.mask_suffix,
                    args.mask_ext,
                )
            valid_cnt[i] = float(vi)

            if args.print_every > 0 and (i + 1) % args.print_every == 0:
                print(f"[{i+1}/{len(img_files)}] vi={vi}  path={img_path}")


    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_index_dir = os.path.dirname(os.path.abspath(args.out_index))
    if out_index_dir:
        os.makedirs(out_index_dir, exist_ok=True)
    out_bucket_dir = os.path.dirname(os.path.abspath(args.out_bucket))
    if out_bucket_dir:
        os.makedirs(out_bucket_dir, exist_ok=True)
    out_bucket_counts_dir = os.path.dirname(os.path.abspath(args.out_bucket_counts))
    if out_bucket_counts_dir:
        os.makedirs(out_bucket_counts_dir, exist_ok=True)

    np.save(args.out, valid_cnt)
    np.save(args.out_index, np.array(img_files, dtype=object))

    valid_cnt_int = np.floor(valid_cnt).astype(np.int32)
    bucket = (valid_cnt_int // int(args.bin_size)).astype(np.int32)
    np.save(args.out_bucket, bucket)

    print(f"Saved valid_cnt: {args.out}  shape={valid_cnt.shape}  min={float(valid_cnt.min()):.2f}  max={float(valid_cnt.max()):.2f}  mean={float(valid_cnt.mean()):.2f}  aug_mode={args.aug_mode}  repeats={args.num_repeats if args.aug_mode=='train' else 1}  agg={args.agg if args.aug_mode=='train' else 'n/a'}")
    print(f"Saved bucket:   {args.out_bucket}  bin_size={args.bin_size}  bucket_min={bucket.min()}  bucket_max={bucket.max()}")
    print(f"Saved index:    {args.out_index}  n={len(img_files)}")

    bincnt = np.bincount(bucket)
    with open(args.out_bucket_counts, "w", encoding="utf-8") as f:
        f.write("bucket_id\tlo\thi\tcount\n")
        for bid, cnt in enumerate(bincnt.tolist()):
            if cnt <= 0:
                continue
            lo = bid * int(args.bin_size)
            hi = (bid + 1) * int(args.bin_size) - 1
            f.write(f"{bid}\t{lo}\t{hi}\t{cnt}\n")


if __name__ == "__main__":
    main()
