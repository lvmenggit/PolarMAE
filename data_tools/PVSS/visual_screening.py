import os
import cv2
import numpy as np
import pickle
import shutil
from glob import glob
from PIL import Image
import imagehash
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm


RAW_IMAGE_DIR = ""

DEDUP_KEEP_DIR = ""
DEDUP_REMOVED_DIR = ""
SAVE_REMOVED_FILES = False  

FEATURE_CACHE_FILE = ""

ROI_SIZE = 256
PHASH_THRESHOLD = 12
COS_THRESHOLD = 0.9   

MIN_THRESH = 12
NUM_WORKERS = 40


SIM_BINS = [
    (0.50, 0.70),
    (0.70, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.94),
    (0.98, 1.01),
]

MAX_SAVE_PER_BIN = 30
SIM_SAVE_DIR = ""
SAVE_SIM_EXAMPLES = False  



POPCOUNT8 = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


def _phash_to_u64(ph) -> np.uint64:

    return np.uint64(int(str(ph), 16))


def _hamming_u64_vec(a: np.ndarray, b: np.uint64) -> np.ndarray:
    x = np.bitwise_xor(a, b)
    xb = x.view(np.uint8).reshape(-1, 8)
    return POPCOUNT8[xb].sum(axis=1)



def extract_fan_roi(img_bgr):
    h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, MIN_THRESH, 255, cv2.THRESH_BINARY)

    binary[:int(h * 0.13), :] = 0  

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()

    roi = img_bgr[top:bottom, left:right]
    if roi.shape[0] < 80 or roi.shape[1] < 80:
        return None

    return cv2.resize(roi, (ROI_SIZE, ROI_SIZE))



def extract_dct_feature(gray, k=32):
    dct = cv2.dct(np.float32(gray))
    feat = dct[:k, :k].flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-6)
    return feat



def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)



def process_one_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    roi = extract_fan_roi(img)
    if roi is None:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    ph = imagehash.phash(Image.fromarray(gray))
    feat = extract_dct_feature(gray)

    rel_path = os.path.relpath(path, RAW_IMAGE_DIR)
    return rel_path, ph, feat



def compute_features(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            enc = pickle.load(f)
        if len(enc) > 0:
            print(f"[1/3] 加载特征缓存 | {len(enc)} 条")
            return enc

    paths = glob(os.path.join(RAW_IMAGE_DIR, "**/*.*"), recursive=True)
    total = len(paths)
    print(f"[1/3] 计算特征 | 图像数: {total}")

    encodings = {}
    processed = 0
    with Pool(NUM_WORKERS) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one_image, paths),
            total=total,
            desc="[1/3] 提取特征",
            dynamic_ncols=True,
        ):
            if res is None:
                continue
            rel_path, ph, feat = res
            encodings[rel_path] = {
                "phash": ph,
                "feat": feat
            }
            processed += 1

    cache_dir = os.path.dirname(os.path.abspath(cache_file))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_file, "wb") as f:
        pickle.dump(encodings, f)

    print(f"  特征完成 | 有效图像: {len(encodings)}\n")
    return encodings


def save_similarity_pair(p1, p2, sim, counter):
    if not SAVE_SIM_EXAMPLES:
        return

    for low, high in SIM_BINS:
        if low <= sim < high:
            key = f"{low:.2f}_{high:.2f}"
            if counter[key] >= MAX_SAVE_PER_BIN:
                return

            save_dir = os.path.join(SIM_SAVE_DIR, key)
            os.makedirs(save_dir, exist_ok=True)


            img1 = cv2.imread(os.path.join(RAW_IMAGE_DIR, p1))
            img2 = cv2.imread(os.path.join(RAW_IMAGE_DIR, p2))
            if img1 is None or img2 is None:
                return


            target_h = 256
            img1 = cv2.resize(
                img1,
                (int(img1.shape[1] * target_h / img1.shape[0]), target_h)
            )
            img2 = cv2.resize(
                img2,
                (int(img2.shape[1] * target_h / img2.shape[0]), target_h)
            )


            pair_img = np.hstack([img1, img2])


            cv2.putText(
                pair_img,
                f"cosine sim = {sim:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            out_name = f"{counter[key]:03d}_sim_{sim:.3f}.png"
            cv2.imwrite(os.path.join(save_dir, out_name), pair_img)

            counter[key] += 1
            return



def deduplicate(encodings):
    print("[2/3] 去重 + 相似度区间采样")

    folder_dict = defaultdict(list)
    for k in encodings.keys():
        folder_dict[os.path.dirname(k)].append(k)

    remove = set()
    sim_counter = defaultdict(int)

    folder_items = sorted(folder_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    for folder, files in tqdm(folder_items, desc="[2/3] 文件夹进度", dynamic_ncols=True):
        files = sorted(files)
        n = len(files)
        if n <= 1:
            continue

        feats = np.stack([encodings[f]["feat"] for f in files]).astype(np.float32, copy=False)
        ph_u64 = np.array([_phash_to_u64(encodings[f]["phash"]) for f in files], dtype=np.uint64)
        removed_local = np.zeros((n,), dtype=bool)

        inner_iter = range(n)
        if n >= 20000:
            inner_iter = tqdm(inner_iter, desc=f"  处理 {os.path.basename(folder) or folder}", leave=False, dynamic_ncols=True)

        for i in inner_iter:
            cur = files[i]
            if cur in remove:
                continue
            if removed_local[i]:
                continue

            if i == 0:
                continue

            prev_idx = np.where(~removed_local[:i])[0]
            if prev_idx.size == 0:
                continue

            hdist = _hamming_u64_vec(ph_u64[prev_idx], ph_u64[i])
            cand_mask = hdist <= int(PHASH_THRESHOLD)
            if not np.any(cand_mask):
                continue
            cand_idx = prev_idx[cand_mask]

            sims = feats[cand_idx] @ feats[i]

            if SAVE_SIM_EXAMPLES:
                for j, sim in zip(cand_idx.tolist(), sims.tolist()):
                    save_similarity_pair(files[j], cur, float(sim), sim_counter)

            if np.any(sims >= float(COS_THRESHOLD)):
                remove.add(cur)
                removed_local[i] = True

    keep = sorted(set(encodings.keys()) - remove)
    print(f"[3/3] 保留 {len(keep)} | 去除 {len(remove)}\n")
    return keep, sorted(remove)

def copy_keep_and_removed(keep_files, remove_files):
    print("[4/4] 保存去重结果")

    os.makedirs(DEDUP_KEEP_DIR, exist_ok=True)
    if SAVE_REMOVED_FILES:
        os.makedirs(DEDUP_REMOVED_DIR, exist_ok=True)


    for idx, rel_path in enumerate(tqdm(keep_files, desc="[4/4] 复制保留样本", dynamic_ncols=True), 1):
        src = os.path.join(RAW_IMAGE_DIR, rel_path)
        dst = os.path.join(DEDUP_KEEP_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except FileNotFoundError:
            continue

    if SAVE_REMOVED_FILES:

        for idx, rel_path in enumerate(remove_files, 1):
            src = os.path.join(RAW_IMAGE_DIR, rel_path)
            dst = os.path.join(DEDUP_REMOVED_DIR, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                continue
            if idx % 500 == 0:
                print(f"  去除样本: 已保存 {idx}/{len(remove_files)}")

    print(" 去重结果已保存")

if __name__ == "__main__":
    encodings = compute_features(FEATURE_CACHE_FILE)
    keep_files, remove_files = deduplicate(encodings)

    save_dir = ""
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "keep_files.pkl"), "wb") as f:
        pickle.dump(keep_files, f)

    if SAVE_REMOVED_FILES:
        with open(os.path.join(save_dir, "remove_files.pkl"), "wb") as f:
            pickle.dump(remove_files, f)
        
    copy_keep_and_removed(keep_files, remove_files)

    if SAVE_SIM_EXAMPLES:
        print("相似度区间样本保存完成")
        print(f"   样本目录: {SIM_SAVE_DIR}")
