# Segmentation/seg/hooks.py
import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class MaskIgnoreForVisHook(Hook):
    """Mask predictions on ignored GT pixels before visualization.

    In mmseg test loop:
      - GT is in data_batch['data_samples']
      - Pred is in outputs (List[SegDataSample])
    """

    def __init__(self, ignore_index: int = 255):
        self.ignore_index = int(ignore_index)

    @torch.no_grad()
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._mask(data_batch, outputs)

    @torch.no_grad()
    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._mask(data_batch, outputs)

    def _mask(self, data_batch, outputs):
        if data_batch is None or outputs is None:
            return
        if 'data_samples' not in data_batch:
            return

        gts = data_batch['data_samples']  # List[SegDataSample] with gt_sem_seg
        preds = outputs                   # List[SegDataSample] with pred_sem_seg

        if not isinstance(gts, (list, tuple)) or not isinstance(preds, (list, tuple)):
            return

        n = min(len(gts), len(preds))
        for i in range(n):
            gt_sample = gts[i]
            pred_sample = preds[i]

            gt_sem_seg = getattr(gt_sample, 'gt_sem_seg', None)
            pred_sem_seg = getattr(pred_sample, 'pred_sem_seg', None)
            if gt_sem_seg is None or pred_sem_seg is None:
                continue

            gt = getattr(gt_sem_seg, 'data', None)
            pred = getattr(pred_sem_seg, 'data', None)
            if not (isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor)):
                continue

            gt2d = gt[0] if gt.dim() == 3 else gt
            pred2d = pred[0] if pred.dim() == 3 else pred

            if gt2d.shape[-2:] != pred2d.shape[-2:]:
                continue

            pred2d[gt2d == self.ignore_index] = self.ignore_index

            # write-back
            if pred.dim() == 3:
                pred[0] = pred2d
            else:
                pred_sem_seg.data = pred2d
