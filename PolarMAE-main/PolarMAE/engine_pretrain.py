import math
from typing import Iterable, Optional, Dict, Tuple, Any

import torch
import torch.distributed as dist

import util.misc as misc
import util.lr_sched as lr_sched


@torch.no_grad()
def _reduce_dict_mean(d: Dict[str, torch.Tensor], device: torch.device):
    """Reduce a dict of scalar tensors across DDP ranks by mean."""
    if not (dist.is_available() and dist.is_initialized()):
        return {k: float(v.item()) for k, v in d.items()}

    keys = sorted(d.keys())
    vals = torch.stack([d[k].detach().to(device=device, dtype=torch.float32) for k in keys], dim=0)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals = vals / float(dist.get_world_size())
    return {k: float(vals[i].item()) for i, k in enumerate(keys)}


def _unpack_batch_for_bucket_mode(
    batch: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Return: (samples, sel_idx, p_polar)

    Expected payload:
        (img, sel_idx)
        (img, sel_idx, p_polar)
    Backward-compatible (treated as NON-bucket):
        (img, sector_mask)
        (img, sector_mask, p_polar)
    """
    payload = batch

    if not (isinstance(payload, (tuple, list)) and (len(payload) == 2 or len(payload) == 3)):
        raise RuntimeError(f"Unexpected payload structure: type={type(payload)}, payload={payload}")

    if len(payload) == 2:
        samples, second = payload
        third = None
    else:
        samples, second, third = payload

    sel_idx = None
    p_polar = None

    # Heuristic:
    # - sel_idx: int tensor, ndim 1/2
    # - sector_mask: float/bool tensor, ndim 3/4
    if third is None:
        if isinstance(second, torch.Tensor) and second.dtype in (torch.int32, torch.int64) and second.ndim in (1, 2):
            sel_idx = second
        else:
            # old (img, sector_mask) case -> not bucket mode
            sel_idx = None
            p_polar = None
    else:
        if isinstance(second, torch.Tensor) and second.dtype in (torch.int32, torch.int64) and second.ndim in (1, 2):
            sel_idx = second
            p_polar = third
        else:
            # old (img, sector_mask, p_polar) case -> not bucket mode
            sel_idx = None
            p_polar = third

    if not isinstance(samples, torch.Tensor):
        raise RuntimeError(f"Unexpected samples type: {type(samples)}")

    return samples, sel_idx, p_polar


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    dataset_train=None,
    args=None,
):
    """
    One training epoch (bucket + sel_idx mode).

    Expected batch payload (bucket mode):
      - (img, sel_idx) or (img, sel_idx, p_polar)

    IMPORTANT:
      - bucket mode下必须提供 sel_idx（[B,K]）
      - p_polar 若提供，必须是 [B,L]，若 dataset 输出 [L] 会在这里自动 expand
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    # ---- verify every 50 epochs ----
    do_verify = (epoch % 50 == 0)

    debug_sum = {
        "K": 0.0,
        "S": 0.0,
        "Mi": 0.0,
        "Mmax": 0.0,
        "masked_total": 0.0,
        "steps": 0.0,
    }
    first_batch_dbg: Optional[Dict[str, float]] = None

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, sel_idx, p_polar = _unpack_batch_for_bucket_mode(batch)
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args,
            )
        samples = samples.to(device, non_blocking=True)
        if sel_idx is None:
            raise RuntimeError(
                "sel_idx is None. In bucketed training, dataset/collate must provide sel_idx. "
                "If you are still in old sector_mask mode, you should call model(samples, sector_mask=...) instead."
            )

        sel_idx = sel_idx.to(device, non_blocking=True).long()

        if sel_idx.ndim == 1:
            sel_idx = sel_idx.unsqueeze(0).expand(samples.shape[0], -1)
        if sel_idx.ndim != 2 or sel_idx.shape[0] != samples.shape[0]:
            raise RuntimeError(f"sel_idx must be [B,K], got {tuple(sel_idx.shape)}, B={samples.shape[0]}")
        K = int(sel_idx.shape[1])
        if K <= 0:
            raise RuntimeError(f"Invalid K from sel_idx: K={K}")

        if p_polar is not None:
            p_polar = p_polar.to(device, non_blocking=True)

            if isinstance(p_polar, torch.Tensor):
                if p_polar.ndim == 1:
                    p_polar = p_polar.unsqueeze(0).expand(samples.shape[0], -1)
                elif p_polar.ndim == 2 and p_polar.shape[0] == 1 and samples.shape[0] > 1:
                    p_polar = p_polar.expand(samples.shape[0], -1)
            else:
                raise RuntimeError(f"Unexpected p_polar type: {type(p_polar)}")

        loss_vec = model(
            samples,
            sel_idx=sel_idx,
            p_polar=p_polar,
            mask_ratio=args.mask_ratio,
            kept_mask_ratio=args.kept_mask_ratio,
        )
        if loss_vec.ndim == 0:
            loss_vec = loss_vec.unsqueeze(0).expand(samples.shape[0])
        if loss_vec.ndim != 1 or loss_vec.shape[0] != samples.shape[0]:
            raise RuntimeError(f"loss_vec must be [B], got {tuple(loss_vec.shape)}, B={samples.shape[0]}")
        loss = loss_vec.mean()

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            raise FloatingPointError(f"Non-finite loss: {loss_value}")
        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=((data_iter_step + 1) % accum_iter == 0),
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

        if do_verify:
            m = model.module if hasattr(model, "module") else model
            dbg = getattr(m, "_debug", None)
            if isinstance(dbg, dict) and len(dbg) > 0:
                if first_batch_dbg is None:
                    first_batch_dbg = {}
                    for k in ("K", "S", "Mi", "Mmax", "masked_total"):
                        if k in dbg:
                            first_batch_dbg[k] = float(dbg[k].item())

                for k in ("K", "S", "Mi", "Mmax", "masked_total"):
                    if k in dbg:
                        debug_sum[k] += float(dbg[k].item())
                debug_sum["steps"] += 1.0

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if do_verify and debug_sum["steps"] > 0:
        t = {k: torch.tensor(v, device=device) for k, v in debug_sum.items()}
        t_mean = _reduce_dict_mean(t, device=device)

        first_out = None
        if first_batch_dbg is not None:
            t1 = {k: torch.tensor(v, device=device) for k, v in first_batch_dbg.items()}
            first_out = _reduce_dict_mean(t1, device=device)

        steps = max(1.0, t_mean["steps"])
        out = {k: (t_mean[k] / steps) for k in t_mean.keys() if k != "steps"}
        out["steps"] = t_mean["steps"]

        if misc.is_main_process():
            if first_out is not None:
                print(f"[VERIFY@epoch={epoch}] first-batch debug:", first_out)
            print(f"[VERIFY@epoch={epoch}] batch-avg debug:", out)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
