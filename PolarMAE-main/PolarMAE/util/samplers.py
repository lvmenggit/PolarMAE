import math
import numpy as np
import torch
import torch.distributed as dist


class BucketDistributedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if not hasattr(dataset, "bucket"):
            raise ValueError("dataset must have attribute `bucket` (np.ndarray[int] of length N).")
        bucket = np.asarray(dataset.bucket)
        if bucket.ndim != 1 or bucket.shape[0] != len(dataset):
            raise ValueError(f"dataset.bucket must be shape [N], got {bucket.shape}, N={len(dataset)}")
        self.bucket = bucket

        # ddp info
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # precompute indices by bucket
        self._bucket2idx = {}
        for i, b in enumerate(self.bucket.tolist()):
            self._bucket2idx.setdefault(int(b), []).append(i)

        # remove tiny buckets if drop_last and cannot form a global batch
        # global batch = batch_size * world_size
        self._buckets = sorted(self._bucket2idx.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # build global batches (each is a list of indices length = batch_size*world_size)
        global_bs = self.batch_size * self.world_size
        all_global_batches = []

        for b in self._buckets:
            idx = self._bucket2idx[b]
            idx = idx.copy()

            if self.shuffle:
                perm = torch.randperm(len(idx), generator=g).tolist()
                idx = [idx[p] for p in perm]

            n = len(idx)
            if self.drop_last:
                n = (n // global_bs) * global_bs
            else:
                # pad to multiple of global_bs (repeat)
                if n % global_bs != 0:
                    pad = global_bs - (n % global_bs)
                    if n > 0:
                        idx = idx + idx[:pad]
                    else:
                        continue
            idx = idx[:n]

            # chunk into global batches
            for s in range(0, len(idx), global_bs):
                all_global_batches.append(idx[s:s + global_bs])

        # shuffle global batch order across buckets (optional but recommended)
        if self.shuffle:
            perm2 = torch.randperm(len(all_global_batches), generator=g).tolist()
            all_global_batches = [all_global_batches[p] for p in perm2]

        # yield this-rank local batch
        for gb in all_global_batches:
            start = self.rank * self.batch_size
            yield gb[start:start + self.batch_size]

    def __len__(self):
        # approximate length (depends on drop_last/padding)
        global_bs = self.batch_size * self.world_size
        total = 0
        for b in self._buckets:
            n = len(self._bucket2idx[b])
            if self.drop_last:
                total += (n // global_bs)
            else:
                total += math.ceil(n / global_bs)
        return total
