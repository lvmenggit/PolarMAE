
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from pathlib import Path


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.data.loader import MultiEpochsDataLoader
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from transformer_utils import handle_flash_attn

import models_mae
import models_polarmae
from engine_pretrain import train_one_epoch

from util.datasets import BucketedSectorMaskPretrainDataset, SectorPairTransform

from util.samplers import BucketDistributedBatchSampler  # type: ignore[reportMissingImports]

from token_selected_smooth import TokenSelect_smooth



def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training (bucketed)", add_help=False)

    # ---------------- basic ----------------
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # ---------------- model ----------------
    parser.add_argument(
        "--model",
        default="mae_vit_base_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--decoder_depth", default=8, type=int, help="depth of decoder")
    parser.add_argument("--mask_ratio", default=0.75, type=float, help="Masking ratio.")
    parser.add_argument(
        "--kept_mask_ratio",
        default=0.25,
        type=float,
        help="Amongst the all tokens, the percentage of the mask that are kept",
    )
    parser.add_argument("--inverse_lr", action="store_true", default=False)
    parser.add_argument("--no_lr_scale", action="store_true", default=False)

    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.set_defaults(find_unused_parameters=True)
    parser.add_argument(
        "--no_find_unused_parameters",
        action="store_false",
        dest="find_unused_parameters",
        help="Disable DDP find_unused_parameters (faster if there are no unused params).",
    )

    # ---------------- optimizer ----------------
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N")

    # ---------------- io/log ----------------
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")

    # ---------------- dataloader ----------------
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--prefetch_factor",
        default=2,
        type=int,
        help="DataLoader prefetch_factor (only used when num_workers>0).",
    )
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--multi_epochs_dataloader", action="store_true")

    # ---------------- distributed ----------------
    parser.add_argument("--world_size", default=8, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    # ---------------- polarmae ----------------
    parser.add_argument("--polarmae", "--selectivemae", dest="polarmae", action="store_true", default=False)
    parser.add_argument("--weight_fm", action="store_true", default=False)
    parser.add_argument("--use_fm", nargs="+", type=int, default=[-1])
    parser.add_argument("--use_input", action="store_true", default=False)
    parser.add_argument("--self_attn", action="store_true", default=False)
    parser.add_argument("--enable_flash_attention2", action="store_true", default=False)

    # ---------------- dataset ----------------
    parser.add_argument("--dataset_path", default="/path/to/train", type=str)

    parser.add_argument("--bucket_path", type=str, default="", help="valid_bucket.npy")
    parser.add_argument("--index_path", type=str, default="", help="valid_cnt_index.npy (可选)")

    parser.add_argument(
        "--sel_k_mode",
        type=str,
        default="bucket_center",
        help="How to convert bucket->K. e.g. bucket_center / bucket_lo / fixed:160",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=16,
        help="bin_size used when generating bucket (must match compute_valid_cnt.py)",
    )
    parser.add_argument(
        "--min_bucket_id",
        type=int,
        default=4,
        help="Filter samples with bucket_id < min_bucket_id (e.g. 4 means K_min≈64 when bin_size=16)",
    )
    parser.add_argument(
        "--bucket_max_tries",
        type=int,
        default=10,
        help="Max retry times for random re-crop when ROI patches < K",
    )
    parser.add_argument(
        "--min_k",
        type=int,
        default=16,
        help="Lower bound for K to avoid tiny ROI batches",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=196,
        help="Upper bound for K (<= L)",
    )


    parser.add_argument("--smooth_scale_epoch", default=20, type=int)


    parser.add_argument("--return_polar", action="store_true", default=True)
    parser.add_argument("--polar_tau", type=float, default=0.1)


    parser.add_argument("--patch_size", type=int, default=16)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    handle_flash_attn(args)


    transform_train = SectorPairTransform(
        args.input_size,
        patch_size=getattr(args, "patch_size", 16),
        return_polar=getattr(args, "return_polar", True),
        tau=getattr(args, "polar_tau", 0.1),
    )

    # ---------------- dataset ----------------
    if not args.bucket_path or not args.index_path:
        raise ValueError(
            "Bucketed pretrain requires --bucket_path and --index_path (valid_bucket.npy / valid_cnt_index.npy)."
        )

    dataset_train = BucketedSectorMaskPretrainDataset(
        root=args.dataset_path,
        pair_transform=transform_train,
        index_npy=args.index_path,
        bucket_npy=args.bucket_path,
        bin_size=args.bin_size,
        patch_size=args.patch_size,  
        roi_tau=getattr(args, "polar_tau", 0.1),
        max_tries=args.bucket_max_tries,
        seed=args.seed,
        min_bucket_id=args.min_bucket_id,
    )
    print(dataset_train)


    bucket_batch_sampler = BucketDistributedBatchSampler(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )
    print("BatchSampler_train =", bucket_batch_sampler)

    # ---------------- dataloader ----------------
    dataloader_cls = MultiEpochsDataLoader if args.multi_epochs_dataloader else torch.utils.data.DataLoader
    dl_kwargs = dict(
        batch_sampler=bucket_batch_sampler, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    if args.num_workers and args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    data_loader_train = dataloader_cls(dataset_train, **dl_kwargs)


    if args.polarmae:
        model = models_polarmae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            decoder_depth=args.decoder_depth,
            roi_tau=getattr(args, "polar_tau", 0.1),
        )
    else:
        model = models_mae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            decoder_depth=args.decoder_depth,
        )

    model.token_select = TokenSelect_smooth(
        expansion_step=[0, 300 - args.smooth_scale_epoch, 300, 600 - args.smooth_scale_epoch, 600],
        keep_rate=[1 - args.mask_ratio] * 5,
        initialization_keep_rate=(1 - args.mask_ratio) / 2,
        expansion_multiple_stage=2,
        smooth_scale_epoch=args.smooth_scale_epoch,
    )

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print("Model = %s" % str(model_without_ddp))
    print("eff_batch_size", eff_batch_size)

    # lr scale
    if args.lr is None:
        base_ratio = args.kept_mask_ratio / args.mask_ratio
        if args.no_lr_scale:
            scale_kmr = 1.0
        elif args.inverse_lr:
            scale_kmr = 1.0 / base_ratio
        else:
            scale_kmr = base_ratio
        args.lr = scale_kmr * args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters
        )
        model_without_ddp = model.module

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # logger
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        print(f"log_dir: {log_writer.log_dir}")
    else:
        log_writer = None

    if args.output_dir and misc.is_main_process():
        args_path = os.path.join(args.output_dir, "args.json")
        with open(args_path, mode="w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Saved args to: {args_path}")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed and hasattr(data_loader_train, "batch_sampler") and hasattr(
            data_loader_train.batch_sampler, "set_epoch"
        ):
            data_loader_train.batch_sampler.set_epoch(epoch)

        model_without_ddp.token_select.update_current_stage(epoch)
        model_without_ddp.token_select.sparse_inference = True

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            dataset_train=None,  
            args=args,
        )

        if args.output_dir:
            if epoch % 50 == 0:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    save_latest_model_only=False,
                )
            if epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    save_latest_model_only=True,
                )

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Log total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(f"Total training time: {total_time_str}\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    assert args.kept_mask_ratio <= args.mask_ratio, "Cannot reconstruct more than what is masked"
    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
