#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert custom per-image json annotations to COCO train/val json files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(""),
        help="Dataset root containing multiple '*output' folders.",
    )
    parser.add_argument(
        "--train-ann",
        type=Path,
        default=None,
        help="Output path for train COCO json. Default: <dataset-root>/train_coco.json",
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=None,
        help="Output path for val COCO json. Default: <dataset-root>/val_coco.json",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--ignore-class",
        action="append",
        default=["关键区域"],
        help="Class names to ignore. Can be used multiple times.",
    )
    parser.add_argument("--min-box-size", type=float, default=1.0, help="Minimum width/height for valid bbox.")
    return parser.parse_args()


def find_json_files(dataset_root: Path) -> List[Path]:
    return sorted(dataset_root.glob("**/json_sorted/*.json"))


def index_images_by_name(dataset_root: Path) -> Dict[str, List[Path]]:
    name_to_paths: Dict[str, List[Path]] = defaultdict(list)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for img_path in dataset_root.glob(f"**/{ext}"):
            name_to_paths[img_path.name].append(img_path)
    return name_to_paths


def resolve_image_path(json_path: Path, image_name: str, image_index: Dict[str, List[Path]]) -> Optional[Path]:
    # Preferred path: sibling folder json_sorted -> jpg with same stem
    candidate = json_path.parent.parent / "jpg" / image_name
    if candidate.exists():
        return candidate

    # Fallback: by basename index
    matches = image_index.get(image_name, [])
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        # Prefer path sharing same '*output' parent
        parent_output = json_path.parent.parent
        for p in matches:
            if p.parent.parent == parent_output:
                return p
        return matches[0]

    return None


def load_custom_annotations(json_path: Path) -> Tuple[Optional[str], List[dict]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    annotations_root = payload.get("annotations", {})
    if not annotations_root:
        return None, []

    # Usually one key per file
    first_key = next(iter(annotations_root.keys()))
    image_meta = annotations_root[first_key]
    annosets = image_meta.get("annosets", [])
    if not annosets:
        return Path(first_key).name, []

    # Use first annoset by default
    det_ann = annosets[0].get("annotations", [])
    return Path(first_key).name, det_ann


def to_xywh(vertex: List[List[float]]) -> Tuple[float, float, float, float]:
    x1, y1 = vertex[0]
    x2, y2 = vertex[1]
    left = float(min(x1, x2))
    top = float(min(y1, y2))
    right = float(max(x1, x2))
    bottom = float(max(y1, y2))
    return left, top, right - left, bottom - top


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    train_ann = args.train_ann or (dataset_root / "train_coco.json")
    val_ann = args.val_ann or (dataset_root / "val_coco.json")

    json_files = find_json_files(dataset_root)
    if not json_files:
        raise RuntimeError(f"No json files found under: {dataset_root}")

    image_index = index_images_by_name(dataset_root)
    ignore_classes = set(args.ignore_class or [])

    image_records = []
    # temp annotations before class-id mapping
    temp_annotations = []
    class_names = set()

    image_id = 1
    ann_id = 1

    for jf in json_files:
        image_name, anns = load_custom_annotations(jf)
        if image_name is None:
            continue

        img_path = resolve_image_path(jf, image_name, image_index)
        if img_path is None:
            print(f"[WARN] Image not found for {jf}")
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            print(f"[WARN] Failed to read image size: {img_path}")
            continue

        file_name = img_path.relative_to(dataset_root).as_posix()
        image_records.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for ann in anns:
            name = ann.get("name")
            vertex = ann.get("vertex")
            if not name or not vertex or len(vertex) != 2:
                continue
            if name in ignore_classes:
                continue

            x, y, w, h = to_xywh(vertex)
            if w < args.min_box_size or h < args.min_box_size:
                continue

            # Clip to image boundary
            x = max(0.0, min(x, width - 1.0))
            y = max(0.0, min(y, height - 1.0))
            w = max(0.0, min(w, width - x))
            h = max(0.0, min(h, height - y))
            if w < args.min_box_size or h < args.min_box_size:
                continue

            class_names.add(name)
            temp_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_name": name,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

        image_id += 1

    if not image_records:
        raise RuntimeError("No valid images found after parsing.")

    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(sorted(class_names))]
    name_to_cid = {c["name"]: c["id"] for c in categories}

    annotations = []
    for ann in temp_annotations:
        cid = name_to_cid.get(ann["category_name"])
        if cid is None:
            continue
        annotations.append(
            {
                "id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": cid,
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"],
                "segmentation": ann["segmentation"],
            }
        )

    # split by image ids
    image_ids = [img["id"] for img in image_records]
    random.Random(args.seed).shuffle(image_ids)
    val_count = int(len(image_ids) * args.val_ratio)
    val_ids = set(image_ids[:val_count])

    train_images = [img for img in image_records if img["id"] not in val_ids]
    val_images = [img for img in image_records if img["id"] in val_ids]

    train_id_set = {img["id"] for img in train_images}
    val_id_set = {img["id"] for img in val_images}

    train_annotations = [ann for ann in annotations if ann["image_id"] in train_id_set]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_id_set]

    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
    }

    train_ann.parent.mkdir(parents=True, exist_ok=True)
    val_ann.parent.mkdir(parents=True, exist_ok=True)

    with train_ann.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)
    with val_ann.open("w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False)

    print("[DONE] COCO files generated")
    print(f"train json: {train_ann}")
    print(f"val json  : {val_ann}")
    print(f"train images: {len(train_images)}, train anns: {len(train_annotations)}")
    print(f"val images  : {len(val_images)}, val anns  : {len(val_annotations)}")
    print(f"num classes : {len(categories)}")
    print("classes:")
    for c in categories:
        print(f"  {c['id']}: {c['name']}")


if __name__ == "__main__":
    main()
