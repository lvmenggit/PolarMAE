import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class IgnoreBlackRegion(BaseTransform):
    """Set near-black image pixels to ignore_index in GT seg map.

    Intended for ultrasound images where large black background regions should
    be excluded from metrics/visualization.
    """

    def __init__(self, threshold: int = 5, ignore_index: int = 255):
        self.threshold = int(threshold)
        self.ignore_index = int(ignore_index)

    def transform(self, results):
        img = results.get('img', None)
        gt_seg_map = results.get('gt_seg_map', None)
        if img is None or gt_seg_map is None:
            return results

        # img could be (H,W,3) or (H,W). LoadImageFromFile typically returns color,
        # but some ultrasound datasets might decode as grayscale.
        if isinstance(img, np.ndarray) and img.ndim == 2:
            black_mask = img <= self.threshold
        else:
            black_mask = np.all(img <= self.threshold, axis=-1)

        gt_seg_map[black_mask] = self.ignore_index
        results['gt_seg_map'] = gt_seg_map
        return results