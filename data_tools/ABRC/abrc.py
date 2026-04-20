import cv2
import numpy as np
import os
import time
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed


def _format_seconds(seconds):
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def extract_sector_smart(img_bgr, min_thresh=10):
    
    h, w = img_bgr.shape[:2]
    total_pixels = h * w
    

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, min_thresh, 255, cv2.THRESH_BINARY)


    ui_cutoff = int(h * 0.13) 
    binary[0:ui_cutoff, :] = 0
    binary[:, :5] = 0
    binary[:, w-5:] = 0


    mid_start = int(w * 0.3)
    mid_end = int(w * 0.7)
    roi_col_sums = np.sum(binary[:, mid_start:mid_end], axis=0)

    gap_thresh = h * 255 * 0.02
    gap_indices = np.where(roi_col_sums < gap_thresh)[0]

    if len(gap_indices) > 5:
        absolute_gap_cols = gap_indices + mid_start
        binary[:, absolute_gap_cols] = 0
        binary[:, min(absolute_gap_cols):max(absolute_gap_cols)] = 0

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)


    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_area = cv2.contourArea(contours[0])

    valid_hulls = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_pixels * 0.03:
            continue
        if area > largest_area * 0.2:
            valid_hulls.append(cv2.convexHull(cnt))

    if not valid_hulls:
        return None, None


    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, valid_hulls, -1, 255, thickness=cv2.FILLED)


    img_clean = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, None

    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()

    pad = 5
    top = max(0, top - pad)
    bottom = min(h, bottom + pad)
    left = max(0, left - pad)
    right = min(w, right + pad)

    cropped_img = img_clean[top:bottom, left:right]
    cropped_mask = (mask[top:bottom, left:right] > 0).astype(np.uint8)

    if cropped_img.shape[0] < 100 or cropped_img.shape[1] < 100:
        return None, None

    return cropped_img, cropped_mask



def _process_one_file(fpath, src_dir, dst_dir, min_thresh=10):
    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    if img is None:
        return 0

    img_out, mask_out = extract_sector_smart(img, min_thresh=min_thresh)
    if img_out is None:
        return 0


    relative_path = os.path.relpath(fpath, src_dir)
    subfolder = os.path.dirname(relative_path)
    save_dir = os.path.join(dst_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(fpath))[0]
    cv2.imwrite(os.path.join(save_dir, name + ".png"), img_out)
    cv2.imwrite(os.path.join(save_dir, name + "_mask.png"), mask_out * 255)
    return 1

def batch_process(src_dir, dst_dir, num_workers=None, min_thresh=10):
    os.makedirs(dst_dir, exist_ok=True)

    exts = ('**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.bmp')
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(src_dir, ext), recursive=True))
    files = sorted(files)
    total_files = len(files)

    if total_files == 0:
        print("没有找到图片，任务结束")
        return

    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = min(32, max(4, cpu_count * 2))

    print(f"找到 {total_files} 张图片，开始处理...")
    print(f"线程数: {num_workers}")

    count_ok = 0
    start_time = time.time()
    progress_interval = 200

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_process_one_file, fpath, src_dir, dst_dir, min_thresh)
            for fpath in files
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                count_ok += fut.result()
            except Exception:

                pass

            if i % progress_interval == 0 or i == total_files:
                elapsed = max(time.time() - start_time, 1e-6)
                speed = i / elapsed
                remaining = total_files - i
                eta_seconds = (remaining / speed) if speed > 0 else None
                print(
                    f"进度: {i}/{total_files} ({i / total_files:.2%}) | "
                    f"有效: {count_ok} | 速度: {speed:.2f} 张/秒 | ETA: {_format_seconds(eta_seconds)}"
                )

    print(f"处理完成，有效图片: {count_ok} 张")

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""
    batch_process(input_folder, output_folder, num_workers=16)
