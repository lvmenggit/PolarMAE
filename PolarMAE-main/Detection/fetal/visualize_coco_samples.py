#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize random COCO annotations.')
    parser.add_argument('--data-root', type=Path, required=True, help='COCO dataset root directory.')
    parser.add_argument('--ann-file', type=str, default='train_coco.json', help='COCO annotation file name under data-root.')
    parser.add_argument('--output-dir', type=Path, default=Path('./coco_vis_samples'), help='Directory to save visualized images.')
    parser.add_argument('--num-samples', type=int, default=30, help='Number of random images to visualize.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-boxes-per-image', type=int, default=200, help='Max boxes drawn per image to avoid clutter.')
    return parser.parse_args()


def load_coco(ann_path: Path) -> dict:
    with ann_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_index(coco: dict):
    images = coco.get('images', [])
    annotations = coco.get('annotations', [])
    categories = coco.get('categories', [])

    imgid_to_info: Dict[int, dict] = {img['id']: img for img in images}
    catid_to_name: Dict[int, str] = {cat['id']: cat['name'] for cat in categories}

    imgid_to_anns: Dict[int, List[dict]] = {img['id']: [] for img in images}
    for ann in annotations:
        img_id = ann.get('image_id')
        if img_id in imgid_to_anns:
            imgid_to_anns[img_id].append(ann)

    return imgid_to_info, catid_to_name, imgid_to_anns


def color_from_cid(cid: int):
    random.seed(cid * 9973)
    return tuple(random.randint(40, 255) for _ in range(3))


def main() -> None:
    args = parse_args()
    ann_path = args.data_root / args.ann_file
    if not ann_path.exists():
        raise FileNotFoundError(f'Annotation file not found: {ann_path}')

    coco = load_coco(ann_path)
    imgid_to_info, catid_to_name, imgid_to_anns = build_index(coco)

    image_ids = list(imgid_to_info.keys())
    if not image_ids:
        raise RuntimeError('No images found in COCO annotation.')

    random.seed(args.seed)
    sample_n = min(args.num_samples, len(image_ids))
    sampled_ids = random.sample(image_ids, sample_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    exported = 0
    skipped = 0

    for img_id in sampled_ids:
        img_info = imgid_to_info[img_id]
        img_path = args.data_root / img_info['file_name']
        if not img_path.exists():
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            skipped += 1
            continue

        draw = ImageDraw.Draw(image)
        anns = imgid_to_anns.get(img_id, [])[: args.max_boxes_per_image]

        for ann in anns:
            bbox = ann.get('bbox', None)
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            cid = ann.get('category_id', -1)
            color = color_from_cid(cid)

            x1, y1 = x, y
            x2, y2 = x + w, y + h
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f'id:{cid}'

            text_bbox = draw.textbbox((x1, y1), label, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            ty1 = max(0, y1 - th - 2)
            draw.rectangle([x1, ty1, x1 + tw + 4, ty1 + th + 2], fill=color)
            draw.text((x1 + 2, ty1 + 1), label, fill=(0, 0, 0), font=font)

        out_name = f"{Path(img_info['file_name']).stem}__vis.jpg"
        out_path = args.output_dir / out_name
        image.save(out_path, quality=95)
        exported += 1

    print('[DONE] Visualization finished')
    print(f'annotation file: {ann_path}')
    print(f'output dir: {args.output_dir.resolve()}')
    print(f'sampled: {sample_n}, exported: {exported}, skipped: {skipped}')


if __name__ == '__main__':
    main()
