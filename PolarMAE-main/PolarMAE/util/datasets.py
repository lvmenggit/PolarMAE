# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import PIL
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils import data
from torch.utils.data import Dataset
import torch.nn.functional as F

from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import cv2
from collections import defaultdict
import random
from typing import List, Iterator, Optional

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None




def build_dataset(is_train, args):
    
    if getattr(args, "use_sector_mask_pretrain", False):
        pair_transform = SectorPairTransform(
            input_size=args.input_size,
            scale=getattr(args, "crop_scale", (0.2, 1.0)),
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            hflip_prob=getattr(args, "hflip", 0.5),

            patch_size=getattr(args, "patch_size", 16),
            return_polar=getattr(args, "return_polar", True),
            polar_dtype=getattr(args, "polar_dtype", torch.float16),
            tau=getattr(args, "polar_tau", 0.1),
            mu=getattr(args, "polar_mu", 0.45),
            sigma=getattr(args, "polar_sigma", 0.20),
            k=getattr(args, "polar_k", 3.0),
            min_comp_patches=getattr(args, "polar_min_comp_patches", 12),
            band_ratio=getattr(args, "polar_band_ratio", 0.15),
        )

        index_npy = getattr(args, "valid_cnt_index_npy", None)
        bucket_npy = getattr(args, "valid_bucket_npy", None)

        if index_npy is not None and bucket_npy is not None:
            dataset = BucketedSectorMaskPretrainDataset(
                root=args.data_path,
                pair_transform=pair_transform,
                index_npy=index_npy,
                bucket_npy=bucket_npy,
                bin_size=getattr(args, "bin_size", 16),
                patch_size=getattr(args, "patch_size", 16),
                roi_tau=getattr(args, "roi_tau", getattr(args, "polar_tau", 0.1)),
                max_tries=getattr(args, "bucket_max_tries", 10),
                seed=getattr(args, "data_seed", 0),
                mask_suffix=getattr(args, "mask_suffix", "_mask"),
                mask_ext=getattr(args, "mask_ext", ".png"),
                min_bucket_id=getattr(args, "min_bucket_id", 1),
                strict_path_match=getattr(args, "strict_path_match", True),
            )
            print(dataset)
            return dataset


        dataset = SectorMaskPretrainDataset(
            root=args.data_path,
            transform=pair_transform,
            mask_suffix=getattr(args, "mask_suffix", "_mask"),
            mask_ext=getattr(args, "mask_ext", ".png"),
        )
        print(dataset)
        return dataset

    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=getattr(args, "color_jitter", 0.4),
            auto_augment=getattr(args, "aa", None),
            interpolation="bicubic",
            re_prob=getattr(args, "reprob", 0.0),
            re_mode=getattr(args, "remode", "pixel"),
            re_count=getattr(args, "recount", 1),
            mean=mean,
            std=std,
        )
        return transform

    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)




class SectorPairTransform:
    def __init__(
        self,
        input_size: int,
        scale=(0.2, 1.0),
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        hflip_prob: float = 0.5,

        patch_size: int = 16,
        return_polar: bool = True,
        polar_dtype: torch.dtype = torch.float16,
        tau: float = 0.1,
        mu: float = 0.45,
        sigma: float = 0.20,
        k: float = 3.0,
        min_comp_patches: int = 12,
        band_ratio: float = 0.15,
    ):
        self.input_size = input_size
        self.scale = scale
        self.mean = mean
        self.std = std
        self.hflip_prob = hflip_prob

        self.patch_size = patch_size
        self.return_polar = return_polar
        self.polar_dtype = polar_dtype

        self.tau = float(tau)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.k = float(k)
        self.min_comp_patches = int(min_comp_patches)
        self.band_ratio = float(band_ratio)

    @torch.no_grad()
    def __call__(self, img: Image.Image, mask: Image.Image):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=self.scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        img = TF.resized_crop(
            img, i, j, h, w, (self.input_size, self.input_size),
            interpolation=InterpolationMode.BICUBIC
        )
        mask = TF.resized_crop(
            mask, i, j, h, w, (self.input_size, self.input_size),
            interpolation=InterpolationMode.NEAREST
        )

        if torch.rand(1).item() < self.hflip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=self.mean, std=self.std)

        mask_t = TF.to_tensor(mask)
        if mask_t.ndim != 3:
            mask_t = mask_t.unsqueeze(0)
        if mask_t.shape[0] != 1:
            mask_t = mask_t[:1]
        mask_t = (mask_t > 0.0).to(dtype=torch.float32)

        if not self.return_polar:
            return img_t, mask_t

        p_polar = self._build_polar_prior_from_mask(mask_t)
        if self.polar_dtype is not None:
            p_polar = p_polar.to(self.polar_dtype)
        return img_t, mask_t, p_polar

    @torch.no_grad()
    def _build_polar_prior_from_mask(self, mask_t: torch.Tensor) -> torch.Tensor:

        assert mask_t.ndim == 3 and mask_t.shape[0] == 1
        H, W = int(mask_t.shape[1]), int(mask_t.shape[2])
        p = int(self.patch_size)
        Gh, Gw = H // p, W // p
        L = Gh * Gw
  
        if Gh <= 0 or Gw <= 0 or Gh * p != H or Gw * p != W:
            out = torch.ones(max(1, L), dtype=torch.float32)
            return out / (out.sum() + 1e-6)


        pooled = F.avg_pool2d(mask_t.unsqueeze(0), kernel_size=p, stride=p)  
        valid_ratio = pooled.flatten(1).squeeze(0).squeeze(0)  

        roi = (valid_ratio.view(Gh, Gw) >= self.tau)  # bool [Gh,Gw]

        roi_u8 = roi.to(torch.uint8).cpu().numpy()
        num, labels = cv2.connectedComponents(roi_u8, connectivity=8)

        if num <= 1:
            out = torch.ones(L, dtype=torch.float32)
            return out / (out.sum() + 1e-6)

        labels_t = torch.from_numpy(labels)
        s_all = torch.zeros((Gh, Gw), dtype=torch.float32) 


        ii = torch.arange(Gh, dtype=torch.float32).view(Gh, 1).expand(Gh, Gw)  
        jj = torch.arange(Gw, dtype=torch.float32).view(1, Gw).expand(Gh, Gw)  
        vr = valid_ratio.view(Gh, Gw)  


        for cid in range(1, num):
            comp = (labels_t == cid)  
            comp_cnt = int(comp.sum().item())
            if comp_cnt < self.min_comp_patches:
                continue  


            row_any = comp.any(dim=1)
            apex_i = int(torch.argmax(row_any.to(torch.int32)).item()) 
            bottom_i = int((Gh - 1) - torch.argmax(torch.flip(row_any.to(torch.int32), dims=[0])).item())  
            denom = float(max(1, bottom_i - apex_i))  


            comp_int = comp.to(torch.int32)
            left = torch.argmax(comp_int, dim=1)
            right = (Gw - 1) - torch.argmax(torch.flip(comp_int, dims=[1]), dim=1)
            empty = ~row_any
            left = left.masked_fill(empty, 0)
            right = right.masked_fill(empty, Gw - 1)
            half_w = ((right - left).to(torch.float32) * 0.5).clamp(min=1.0)  

   
            band_hi = min(Gh - 1, apex_i + int(self.band_ratio * Gh))
            band = torch.zeros((Gh, Gw), dtype=torch.bool)
            band[apex_i:band_hi + 1, :] = True
            band = band & comp

            w = (vr * band.to(torch.float32))
            wsum = float(w.sum().item())
            if wsum < 1e-6:
                w = comp.to(torch.float32)
                wsum = float(w.sum().item())
            center_j = float((w * jj).sum().item() / (wsum + 1e-6))

            i = ii[comp]
            j = jj[comp]


            r_norm = ((i - float(apex_i)) / denom).clamp(0.0, 1.0)  
            hw = half_w[i.to(torch.long)]  
            th_norm = ((j - center_j) / hw).clamp(-1.0, 1.0)  

            f_r = torch.exp(- (r_norm - self.mu) ** 2 / (2.0 * (self.sigma ** 2) + 1e-6))

            g_th = 1.0 - th_norm.abs().pow(self.k)

            gate = ((vr[comp] - self.tau) / (1.0 - self.tau + 1e-6)).clamp(0.0, 1.0)
            s = gate * (f_r + g_th)

            tmp = torch.zeros((Gh, Gw), dtype=torch.float32)
            tmp[comp] = s
            s_all = torch.maximum(s_all, tmp)

        s_flat = s_all.reshape(-1).clamp_min(0.0)
        if float(s_flat.sum().item()) < 1e-6:
            s_flat = torch.ones_like(s_flat)
        p_polar = s_flat / (s_flat.sum() + 1e-6)
        return p_polar




class SectorMaskPretrainDataset(data.Dataset):
    def __init__(self, root: str, transform: SectorPairTransform,
                 mask_suffix: str = "_mask", mask_ext: str = ".png"):
        self.root = root
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.mask_ext = mask_ext

        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lower = fn.lower()
                if not lower.endswith(exts):
                    continue
                stem, _ = os.path.splitext(fn)
                if stem.endswith(self.mask_suffix):
                    continue
                files.append(os.path.join(dirpath, fn))
        self.files = sorted(files)
        print(f"创建 SectorMaskPretrainDataset，包含 {len(self.files)} 个样本")

    def __len__(self):
        return len(self.files)

    def _mask_path(self, img_path: str) -> str:
        base, _ = os.path.splitext(img_path)
        return base + f"{self.mask_suffix}{self.mask_ext}"

    def __getitem__(self, i: int):
        n = len(self.files)
        for _ in range(10):
            img_path = self.files[i % n]
            mask_path = self._mask_path(img_path)
            try:
                img = Image.open(img_path).convert("RGB")
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert("L")
                else:
                    mask = Image.new("L", img.size, color=255)

                out = self.transform(img, mask)
                if isinstance(out, (tuple, list)) and len(out) == 3:
                    return out  # (img_t, mask_t, p_polar)
                else:
                    return out  # (img_t, mask_t)

            except Exception as e:
                print(f"加载配对 {img_path} / {mask_path} 时出错：{e}")
                i += 1

        dummy_img = torch.zeros(3, self.transform.input_size, self.transform.input_size, dtype=torch.float32)
        dummy_mask = torch.ones(1, self.transform.input_size, self.transform.input_size, dtype=torch.float32)
        if getattr(self.transform, "return_polar", False):
            p = getattr(self.transform, "patch_size", 16)
            L = (self.transform.input_size // p) * (self.transform.input_size // p)
            dummy_p = torch.ones(L, dtype=torch.float32)
            dummy_p = dummy_p / (dummy_p.sum() + 1e-6)
            if getattr(self.transform, "polar_dtype", None) is not None:
                dummy_p = dummy_p.to(self.transform.polar_dtype)
            return dummy_img, dummy_mask, dummy_p
        return dummy_img, dummy_mask



def _mask_to_valid_patch_idx(mask_t: torch.Tensor, patch_size: int, roi_tau: float) -> torch.Tensor:
    assert mask_t.ndim == 3 and mask_t.shape[0] == 1
    pooled = F.avg_pool2d(mask_t.unsqueeze(0), kernel_size=patch_size, stride=patch_size)  # [1,1,Gh,Gw]
    valid = (pooled[0, 0] >= roi_tau).flatten()
    return torch.nonzero(valid, as_tuple=False).flatten().long()

def _sample_k_from_roi(valid_idx: torch.Tensor, K: int, g: torch.Generator) -> torch.Tensor:
    vi = int(valid_idx.numel())
    if vi <= 0:
        raise RuntimeError("valid_idx empty. Check masks or roi_tau.")
    if vi >= K:
        perm = torch.randperm(vi, generator=g, device=valid_idx.device)
        return valid_idx[perm[:K]]
    need = K - vi
    extra = torch.randint(0, vi, (need,), generator=g, device=valid_idx.device)
    return torch.cat([valid_idx, valid_idx[extra]], dim=0)

class BucketedSectorMaskPretrainDataset(Dataset):
    """
    预训练 Dataset（同桶 batch + 固定 K 的配套 Dataset）：
      - 离线：每个样本有 bucket_id（valid_bucket.npy），对应一个固定 K = bucket_id * bin_size（桶下界）
      - 在线训练：__getitem__ 内做训练同款随机增强（由 SectorPairTransform 完成）
      - 每次增强后，根据 mask 计算 vi_aug；若 vi_aug < K：
            重裁剪（重新跑 transform）最多 max_tries 次；
            若仍不足：ROI 内重复采样补齐到 K（不引入 ROI 外）
      - 可选返回 p_polar：如果你的 SectorPairTransform.return_polar=True，会返回 (img,mask,p_polar)

        返回（与 engine_pretrain.py 的 bucket mode 对齐）：
            - 若 transform 返回 polar： (img_t, sel_idx, p_polar)
            - 若不返回 polar：        (img_t, sel_idx)

        说明：mask_t 仅用于在 dataset 内计算 ROI 与 p_polar，不再向上游返回。
    """

    def __init__(
        self,
        root: str,
        pair_transform,                
        index_npy: str,                 # valid_cnt_index.npy
        bucket_npy: str,                # valid_bucket.npy
        bin_size: int = 16,
        patch_size: int = 16,
        roi_tau: float = 0.1,
        max_tries: int = 10,
        seed: int = 0,
        mask_suffix: str = "_mask",
        mask_ext: str = ".png",
        min_bucket_id: int = 5,         
        strict_path_match: bool = True, 
    ):
        super().__init__()
        self.root = root
        self.bin_size = int(bin_size)
        self.patch_size = int(patch_size)
        self.roi_tau = float(roi_tau)
        self.max_tries = int(max_tries)
        self.seed = int(seed)
        self.min_bucket_id = int(min_bucket_id)

   
        self.base = SectorMaskPretrainDataset(
            root=root,
            transform=pair_transform,
            mask_suffix=mask_suffix,
            mask_ext=mask_ext,
        )


        self.paths = np.load(index_npy, allow_pickle=True)
        self.bucket = np.load(bucket_npy).astype(np.int32)
        assert len(self.paths) == len(self.bucket), "index_npy and bucket_npy length mismatch"


        if self.min_bucket_id > 1:
            keep = self.bucket >= np.int32(self.min_bucket_id)
            before = int(len(self.paths))
            kept = int(keep.sum())
            if kept <= 0:
                raise RuntimeError(
                    f"After filtering with min_bucket_id={self.min_bucket_id}, no samples remain. "
                    f"Consider lowering min_bucket_id or regenerating buckets."
                )
            if kept < before:
                self.paths = self.paths[keep]
                self.bucket = self.bucket[keep]
                print(
                    f"BucketedSectorMaskPretrainDataset: filtered low buckets "
                    f"min_bucket_id={self.min_bucket_id}, kept {kept}/{before} samples."
                )


        self._path2base = {p: i for i, p in enumerate(self.base.files)}

        if strict_path_match:

            miss = 0
            for p in self.paths[: min(1000, len(self.paths))]:
                if str(p) not in self._path2base:
                    miss += 1
            if miss > 0:
                raise RuntimeError(
                    f"Path mismatch between index_npy and dataset scan. "
                    f"First 1000 samples missing={miss}. "
                    f"Ensure index_npy was generated by scanning the same root with same path format."
                )

    def __len__(self):
        return len(self.paths)

    def _K_from_bucket(self, bid: int) -> int:

        return int(bid) * self.bin_size

    def __getitem__(self, i: int):
        path = str(self.paths[i])
        bid = int(self.bucket[i])

        if bid < self.min_bucket_id:
            bid = self.min_bucket_id

        K = self._K_from_bucket(bid)
        if K <= 0:
            K = self.bin_size

        base_i = self._path2base.get(path, None)
        if base_i is None:

            base_i = 0

        g = torch.Generator()
        g.manual_seed(self.seed + i * 10007)

        last = None
        for _ in range(self.max_tries):
            out = self.base[base_i]  

            if isinstance(out, (tuple, list)) and len(out) == 3:
                img_t, mask_t, p_polar = out
            else:
                img_t, mask_t = out
                p_polar = None


            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)
            if mask_t.shape[0] != 1:
                mask_t = mask_t[:1]

            mask_t = (mask_t > 0.0).to(dtype=torch.float32)

            grid_h = int(mask_t.shape[-2]) // int(self.patch_size)
            grid_w = int(mask_t.shape[-1]) // int(self.patch_size)
            L = int(grid_h * grid_w)
            if L <= 0:
                raise RuntimeError(
                    f"Invalid patch grid size L={L} from mask shape={tuple(mask_t.shape)} "
                    f"and patch_size={self.patch_size}."
                )
            K_eff = int(min(int(K), L))
            if K_eff <= 0:
                K_eff = 1

            if p_polar is not None:
                s = float(p_polar.sum())
                if (not torch.isfinite(p_polar).all()) or (s <= 0.0):
                    p_polar = torch.ones(L, dtype=p_polar.dtype, device=p_polar.device)
                    p_polar = p_polar / (p_polar.sum() + 1e-6)

            valid_idx = _mask_to_valid_patch_idx(mask_t, self.patch_size, self.roi_tau)
            last = (img_t, mask_t, p_polar, valid_idx)

            if int(valid_idx.numel()) >= K_eff:
                sel_idx = _sample_k_from_roi(valid_idx, K_eff, g)
                if p_polar is None:
                    return img_t, sel_idx
                return img_t, sel_idx, p_polar


        if last is None:
            raise RuntimeError("Dataset retry loop produced no samples (last is None).")

        img_t, last_mask_t, p_polar, valid_idx = last

        grid_h = int(last_mask_t.shape[-2]) // int(self.patch_size)
        grid_w = int(last_mask_t.shape[-1]) // int(self.patch_size)
        L = int(grid_h * grid_w)
        if L <= 0:
            raise RuntimeError(
                f"Invalid patch grid size L={L} from mask shape={tuple(last_mask_t.shape)} "
                f"and patch_size={self.patch_size}."
            )
        K_eff = int(min(int(K), L))
        if K_eff <= 0:
            K_eff = 1

        if int(valid_idx.numel()) <= 0:
            sel_idx = torch.randint(0, L, (K_eff,), generator=g, device=valid_idx.device)
        else:
            sel_idx = _sample_k_from_roi(valid_idx, K_eff, g)
        if p_polar is None:
            return img_t, sel_idx
        return img_t, sel_idx, p_polar