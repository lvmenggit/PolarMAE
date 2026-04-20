import os
import shutil
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image, ImageFile
from tqdm import tqdm
from collections import defaultdict
import cv2
import hashlib
import logging

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


RAW_IMAGE_DIR = ""
DEDUP_KEEP_DIR = ""
DEDUP_REMOVED_DIR = ""
FEATURE_CACHE_FILE = ""

SIM_SAVE_DIR = ""
MAP_FILE = os.path.join(SIM_SAVE_DIR, "pairs.tsv")

LOCAL_PRETRAINED_DIR = "./pretrained/medclip-vit"

BATCH_SIZE_PER_GPU = 256
NUM_WORKERS = 16
COMPUTE_CHUNK_SIZE = 200

COS_THRESHOLD = 0.985

SIM_BINS = [
    (0.60, 0.80), (0.80, 0.90), (0.90, 0.95),
    (0.95, 0.97), (0.97, 0.98), (0.98, 0.985), (0.985, 1.01),
]
MAX_SAVE_PER_BIN = 50


SAVE_PREVIEW_JPG = True
SAVE_STITCHED_PAIR = True
STITCH_HEIGHT = 512 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_COUNT = torch.cuda.device_count()
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * max(1, GPU_COUNT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



def extract_fan_roi(img_pil: Image.Image) -> Image.Image:
    try:
        cv2.setNumThreads(0)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)


        binary[:int(h * 0.13), :] = 0


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts_info = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
        if not contours:
            return img_pil.resize((224, 224))

        cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return img_pil.resize((224, 224))

        top, bottom = ys.min(), ys.max()
        left, right = xs.min(), xs.max()

        roi = img_bgr[top:bottom, left:right]
        roi = cv2.resize(roi, (224, 224))
        return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    except Exception:
        return img_pil.resize((224, 224))


class MedImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            processed_img = extract_fan_roi(img)
            return processed_img, path
        except Exception:
            return None, path


def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)


def _move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out



def load_medclip_model():
    logger.info(f"[1/4] 加载 MedCLIP 模型 | GPU 数量: {GPU_COUNT}")

    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    try:
        if LOCAL_PRETRAINED_DIR and os.path.isdir(LOCAL_PRETRAINED_DIR):
            logger.info(f"load model weight from: {LOCAL_PRETRAINED_DIR}")
            model.from_pretrained(LOCAL_PRETRAINED_DIR)
        else:
            model.from_pretrained()
    except TypeError:
        model.from_pretrained()

    model.eval()

    if DEVICE == "cuda" and GPU_COUNT > 1:
        logger.info(f"  ⚡ 启用多卡并行 (DataParallel) on {GPU_COUNT} GPUs")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)
    processor = MedCLIPProcessor()
    return model, processor


@torch.no_grad()
def _forward_get_img_embeds(model, inputs, force_single_gpu=False):
    if DEVICE != "cuda" or GPU_COUNT <= 1:
        outputs = model(**inputs)
    else:
        if force_single_gpu and isinstance(model, nn.DataParallel):
            m = model.module
            inputs0 = _move_to_device(inputs, "cuda:0")
            outputs = m(**inputs0)
        else:
            outputs = model(**inputs)

    if isinstance(outputs, dict):
        return outputs["img_embeds"]
    return outputs.img_embeds


def compute_features(model, processor, cache_file):
    if os.path.exists(cache_file):
        logger.info(f"[2/4] 发现特征缓存，直接加载: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info(f"[2/4] 开始提取特征 (CPU核数:{NUM_WORKERS}, Batch:{TOTAL_BATCH_SIZE})...")

    paths = glob(os.path.join(RAW_IMAGE_DIR, "**/*.*"), recursive=True)
    valid_paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    valid_paths.sort()
    logger.info(f"  待处理图片总数: {len(valid_paths)}")

    dataset = MedImageDataset(valid_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=TOTAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    encodings = {}

    for batch_imgs, batch_paths in tqdm(dataloader, desc="Extracting"):
        if batch_imgs is None:
            continue

        bs = len(batch_imgs)
        try:
            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
            if "input_ids" not in inputs:
                dummy = processor(text=[""] * bs, return_tensors="pt", padding=True)
                inputs.update(dummy)

            need_single = (DEVICE == "cuda" and GPU_COUNT > 1 and (bs % GPU_COUNT != 0))
            if not need_single:
                inputs = _move_to_device(inputs, DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                img_embeds = _forward_get_img_embeds(model, inputs, force_single_gpu=need_single)

            img_embeds = img_embeds.float()
            img_embeds = img_embeds / (img_embeds.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            img_embeds = torch.nan_to_num(img_embeds)

            img_embeds = img_embeds.detach().cpu().numpy()

            for path, embed in zip(batch_paths, img_embeds):
                rel_path = os.path.relpath(path, RAW_IMAGE_DIR) 
                encodings[rel_path] = embed

        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            import traceback
            traceback.print_exc()

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(encodings, f)

    return encodings



def _read_for_preview(src_path: str) -> np.ndarray | None:
   
 
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype != np.uint8:
        vmin, vmax = np.percentile(gray, (1, 99))
        if vmax <= vmin:
            vmax = vmin + 1
        gray8 = np.clip((gray - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)
    else:
        gray8 = gray

    bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    return bgr


def _save_preview(src_path: str, dst_jpg: str) -> None:
    img = _read_for_preview(src_path)
    if img is None:
        return
    cv2.imwrite(dst_jpg, img)


def _save_stitched_pair(src1: str, src2: str, out_jpg: str, sim: float,
                        rel1: str, rel2: str, height: int = 512) -> None:
    a = _read_for_preview(src1)
    b = _read_for_preview(src2)
    if a is None or b is None:
        return

    def resize_to_h(img, h):
        H, W = img.shape[:2]
        if H == 0:
            return img
        new_w = max(1, int(W * (h / H)))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    a = resize_to_h(a, height)
    b = resize_to_h(b, height)

    gap = 12
    canvas = np.zeros((height, a.shape[1] + gap + b.shape[1], 3), dtype=np.uint8)
    canvas[:, :a.shape[1]] = a
    canvas[:, a.shape[1] + gap:] = b


    txt1 = f"sim={sim:.4f}"

    def short(s, n=70):
        return s if len(s) <= n else ("..." + s[-(n-3):])

    line2 = "A: " + short(rel1)
    line3 = "B: " + short(rel2)

    cv2.putText(canvas, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, line2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)
    cv2.imwrite(out_jpg, canvas)


def save_sample_pair(rel_p1, rel_p2, sim, counter):
   
    for low, high in SIM_BINS:
        if low <= sim < high:
            key = f"{low:.2f}_{high:.2f}"
            if counter[key] >= MAX_SAVE_PER_BIN:
                return

            base_dir = os.path.join(SIM_SAVE_DIR, key)
            a_dir = os.path.join(base_dir, "A")
            b_dir = os.path.join(base_dir, "B")
            pair_dir = os.path.join(base_dir, "pairs")

            dst1 = os.path.join(a_dir, rel_p1)
            dst2 = os.path.join(b_dir, rel_p2)
            os.makedirs(os.path.dirname(dst1), exist_ok=True)
            os.makedirs(os.path.dirname(dst2), exist_ok=True)

            src1 = os.path.join(RAW_IMAGE_DIR, rel_p1)
            src2 = os.path.join(RAW_IMAGE_DIR, rel_p2)

            tag = hashlib.md5((rel_p1 + "||" + rel_p2).encode()).hexdigest()[:8]
            if os.path.exists(dst1):
                root, ext = os.path.splitext(dst1)
                dst1 = f"{root}__{tag}{ext}"
            if os.path.exists(dst2):
                root, ext = os.path.splitext(dst2)
                dst2 = f"{root}__{tag}{ext}"
                os.makedirs(os.path.dirname(dst2), exist_ok=True)

            shutil.copy2(src1, dst1)
            shutil.copy2(src2, dst2)

            if SAVE_PREVIEW_JPG:
                p1_prev = os.path.splitext(dst1)[0] + "__preview.jpg"
                p2_prev = os.path.splitext(dst2)[0] + "__preview.jpg"
                try:
                    _save_preview(src1, p1_prev)
                    _save_preview(src2, p2_prev)
                except Exception:
                    pass

            if SAVE_STITCHED_PAIR:
                pair_name = f"{sim:.4f}__{tag}__pair.jpg"
                pair_path = os.path.join(pair_dir, pair_name)
                try:
                    _save_stitched_pair(src1, src2, pair_path, sim, rel_p1, rel_p2, height=STITCH_HEIGHT)
                except Exception:
                    pass

            os.makedirs(SIM_SAVE_DIR, exist_ok=True)
            with open(MAP_FILE, "a", encoding="utf-8") as f:
                f.write(f"{sim:.6f}\t{rel_p1}\t{rel_p2}\t{dst1}\t{dst2}\n")

            counter[key] += 1
            return


def bins_full(counter):
    for low, high in SIM_BINS:
        if counter[f"{low:.2f}_{high:.2f}"] < MAX_SAVE_PER_BIN:
            return False
    return True


@torch.no_grad()
def fill_bins_from_sim(sim, q_global_idx, g_global_idx, paths, counter,
                      topk=3000, randm=20000, fallback_limit_per_bin=20000):
    if sim.numel() == 0 or bins_full(counter):
        return

    device = sim.device
    R, C = sim.shape
    flat = sim.flatten()

    if topk > 0 and not bins_full(counter):
        k = min(topk, flat.numel())
        vals, idx = torch.topk(flat, k=k)
        for v, idv in zip(vals.tolist(), idx.tolist()):
            if bins_full(counter):
                break
            r = idv // C
            c = idv % C
            save_sample_pair(paths[q_global_idx[r]], paths[g_global_idx[c]], float(v), counter)

    if randm > 0 and not bins_full(counter):
        m = min(randm, flat.numel())
        idx = torch.randint(0, flat.numel(), (m,), device=device)
        vals = flat[idx].tolist()
        idx = idx.tolist()
        for v, idv in zip(vals, idx):
            if bins_full(counter):
                break
            r = idv // C
            c = idv % C
            save_sample_pair(paths[q_global_idx[r]], paths[g_global_idx[c]], float(v), counter)

    if bins_full(counter):
        return

    for low, high in SIM_BINS:
        key = f"{low:.2f}_{high:.2f}"
        need = MAX_SAVE_PER_BIN - counter[key]
        if need <= 0:
            continue

        mask = (flat >= low) & (flat < high)
        idx_all = torch.nonzero(mask, as_tuple=False).flatten()
        if idx_all.numel() == 0:
            continue

        if idx_all.numel() > fallback_limit_per_bin:
            perm = torch.randperm(idx_all.numel(), device=device)[:fallback_limit_per_bin]
            idx_all = idx_all[perm]

        take = min(need, idx_all.numel())
        perm = torch.randperm(idx_all.numel(), device=device)[:take]
        idx_pick = idx_all[perm].tolist()

        for idv in idx_pick:
            r = idv // C
            c = idv % C
            v = float(flat[idv].item())
            save_sample_pair(paths[q_global_idx[r]], paths[g_global_idx[c]], v, counter)

        if bins_full(counter):
            return


def deduplicate_global_gpu(encodings):
    logger.info("[3/4] 全局语义去重 (GPU 分块加速)...")

    paths = sorted(list(encodings.keys()))  
    if not paths:
        return [], []

    logger.info("  正在将特征加载到 GPU...")
    feats = torch.tensor(np.stack([encodings[p] for p in paths]),
                         device=DEVICE, dtype=torch.float32)

    N = len(paths)
    remove_indices = set()
    sim_counter = defaultdict(int)

    os.makedirs(SIM_SAVE_DIR, exist_ok=True)
    if os.path.exists(MAP_FILE):
        os.remove(MAP_FILE)

    for i in tqdm(range(0, N, COMPUTE_CHUNK_SIZE), desc="Deduplicating"):
        end_i = min(i + COMPUTE_CHUNK_SIZE, N)

        block_idx = [idx for idx in range(i, end_i) if idx not in remove_indices]
        if not block_idx:
            continue

        block_feats = feats[block_idx]

        sim_in = torch.mm(block_feats, block_feats.T)
        tri = torch.triu(torch.ones_like(sim_in, dtype=torch.bool), diagonal=1)
        sim_tri = sim_in.masked_fill(~tri, -1)

        if not bins_full(sim_counter):
            fill_bins_from_sim(sim_tri, block_idx, block_idx, paths, sim_counter,
                              topk=1500, randm=6000, fallback_limit_per_bin=10000)

        _, c_in = torch.where((sim_in >= COS_THRESHOLD) & tri)
        for c in c_in.detach().cpu().tolist():
            remove_indices.add(block_idx[c])

        block_idx = [idx for idx in block_idx if idx not in remove_indices]
        if not block_idx:
            continue
        block_feats = feats[block_idx]

        if end_i >= N:
            break

        gallery_start = end_i
        gallery_idx = list(range(gallery_start, N))
        gallery_feats = feats[gallery_start:]

        sim = torch.mm(block_feats, gallery_feats.T)

        if not bins_full(sim_counter):
            fill_bins_from_sim(sim, block_idx, gallery_idx, paths, sim_counter,
                              topk=3000, randm=20000, fallback_limit_per_bin=20000)

        _, dup_cols = torch.where(sim >= COS_THRESHOLD)
        for c in dup_cols.detach().cpu().tolist():
            remove_indices.add(gallery_start + int(c))

    remove_files = [paths[idx] for idx in sorted(remove_indices)]
    keep_files = [p for idx, p in enumerate(paths) if idx not in remove_indices]

    for low, high in SIM_BINS:
        key = f"{low:.2f}_{high:.2f}"
        print(f"  {key}: {sim_counter[key]}")
    zeros = [f"{low:.2f}_{high:.2f}" for low, high in SIM_BINS if sim_counter[f"{low:.2f}_{high:.2f}"] == 0]
    if zeros:
        print(f"\n以下区间采样为0：{zeros}（通常说明数据里几乎没有落在该区间的相似对）")

    print(f"\n保留: {len(keep_files)} | 去除: {len(remove_files)}")
    print(f"✅ 样本对照(含拼接图)目录: {SIM_SAVE_DIR}")
    print(f"✅ 映射表: {MAP_FILE}")
    return keep_files, remove_files


def save_results(keep_files, remove_files):
    logger.info("[4/4] 保存去重结果 (开始文件复制)...")

    os.makedirs(os.path.dirname(FEATURE_CACHE_FILE), exist_ok=True)
    with open(os.path.join(os.path.dirname(FEATURE_CACHE_FILE), "keep_medclip_global.pkl"), "wb") as f:
        pickle.dump(keep_files, f)
    with open(os.path.join(os.path.dirname(FEATURE_CACHE_FILE), "remove_medclip_global.pkl"), "wb") as f:
        pickle.dump(remove_files, f)

    logger.info(f"  正在复制 {len(keep_files)} 个保留文件到: {DEDUP_KEEP_DIR}")
    for rel_path in tqdm(keep_files, desc="Copying Keep"):
        src = os.path.join(RAW_IMAGE_DIR, rel_path)
        dst = os.path.join(DEDUP_KEEP_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            logger.error(f"Copy failed {src}: {e}")

    logger.info(f"  正在复制 {len(remove_files)} 个重复文件到: {DEDUP_REMOVED_DIR}")
    for rel_path in tqdm(remove_files, desc="Copying Remove"):
        src = os.path.join(RAW_IMAGE_DIR, rel_path)
        dst = os.path.join(DEDUP_REMOVED_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            logger.error(f"Copy failed {src}: {e}")

    print("✅ 全局语义去重完成")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    model, processor = load_medclip_model()
    encodings = compute_features(model, processor, FEATURE_CACHE_FILE)
    keep_files, remove_files = deduplicate_global_gpu(encodings)
    save_results(keep_files, remove_files)
