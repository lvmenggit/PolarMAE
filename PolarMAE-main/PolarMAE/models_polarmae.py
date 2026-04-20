# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import partial
import math
import time
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformer_utils import Block, CrossAttentionBlock, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
from token_selected_smooth import TokenSelect_smooth as TokenSelect


class HOGLayerC(nn.Module):
    def __init__(
        self,
        nbins: int = 9,
        pool: int = 8,
        gaussian_window: int = 16,
        norm_out: bool = False,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        self.in_channels = in_channels

        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = self.get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)
        self.norm_out = norm_out

    def get_gkern(self, kernlen: int, std: int) -> torch.Tensor:
        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        gkern1d = _gaussian_fn(kernlen, std)
        gkern2d = gkern1d[:, None] * gkern1d[None, :]
        return gkern2d / gkern2d.sum()

    def _reshape(self, hog_feat: torch.Tensor, Gh: int, Gw: int) -> torch.Tensor:
        """
        hog_feat: [B, C, nbins, H', W'] after pooling
        return:   [B, Gh*Gw, ...] then will be reduced to [B, Gh*Gw]
        """
        # flatten C and nbins into channel-like dim
        hog_feat = hog_feat.flatten(1, 2)  # [B, C*nbins, H', W']

        # map H',W' -> Gh,Gw by pooling windows (assume divisible)
        H2, W2 = hog_feat.shape[-2], hog_feat.shape[-1]
        assert H2 % Gh == 0 and W2 % Gw == 0, f"HOG grid {H2}x{W2} not divisible by token grid {Gh}x{Gw}"
        sh, sw = H2 // Gh, W2 // Gw

        hog_feat = (
            hog_feat.permute(0, 2, 3, 1)         # [B, H2, W2, Cnb]
            .unfold(1, sh, sh)                   # [B, Gh, W2, Cnb, sh]
            .unfold(2, sw, sw)                   # [B, Gh, Gw, Cnb, sh, sw]
            .flatten(1, 2)                       # [B, Gh*Gw, Cnb, sh, sw]
            .flatten(2)                          # [B, Gh*Gw, Cnb*sh*sw]
        )
        return hog_feat

    @torch.no_grad()
    def forward(self, x: torch.Tensor, Gh: int, Gw: int) -> torch.Tensor:
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=self.in_channels)
        gy = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=self.in_channels)

        norm = torch.stack([gx, gy], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx, gy)
        phase = phase / self.pi * self.nbins

        b, c, h, w = norm.shape
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)

        phase = phase.view(b, c, 1, h, w)
        norm = norm.view(b, c, 1, h, w)

        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, f"h {h} gw {self.gaussian_window}"
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm = norm * temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm)

        # Optimization: Use avg_pool2d instead of unfold + sum for speed and memory efficiency
        # out: [B, C, nbins, H, W] -> Flatten first 3 dims -> [B*C*nb, 1, H, W]
        b, c, nb, h, w = out.shape
        out = out.reshape(b * c * nb, 1, h, w)
        out = F.avg_pool2d(out, kernel_size=self.pool, stride=self.pool) * (self.pool * self.pool)
        out = out.reshape(b, c, nb, out.shape[2], out.shape[3]) # Reshape back

        if self.norm_out:
            out = F.normalize(out, p=2, dim=2)

        out_1d = self._reshape(out, Gh, Gw)
        mean = out_1d.mean(dim=-1, keepdim=True)
        var = out_1d.var(dim=-1, keepdim=True)
        out_1d = (out_1d - mean) / (var + 1e-6) ** 0.5
        out = out_1d.mean(dim=-1)
        return out


class MaskedAutoencoderViT(nn.Module):
    """
    Selective MAE + ROI-sector masking + mean(valid) align + ROI不足时按分布补齐。

    方案1（本文件实现）：
      - 不在模型里计算极坐标（不使用 cv2 / .cpu().numpy()）
      - dataset 侧预先算好 p_polar: [N,L]，模型侧只负责融合
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        weight_fm=False,
        use_fm=[-1],
        use_input=False,
        self_attn=False,
        mask_cap: int | None = 128,
        lam_polar: float = 0.3,   # 你原来 lam=0.3
        roi_tau: float = 0.1,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.norm_pix_loss = norm_pix_loss
        self.mask_cap = mask_cap
        self.lam_polar = float(lam_polar)
        self.roi_tau = float(roi_tau)

        self.token_select = TokenSelect()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed1 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(
                decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio,
                qkv_bias=True, qk_scale=None, norm_layer=norm_layer, self_attn=self_attn
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.hog = HOGLayerC(nbins=9, pool=8, norm_out=True, in_channels=in_chans)
        self.softmax = nn.Softmax(dim=-1)

        self.initialize_weights()

        self.use_input = use_input
        if len(use_fm) == 1 and use_fm[0] == -1:
            self.use_fm = list(range(depth))
        else:
            self.use_fm = [i if i >= 0 else depth + i for i in use_fm]

        self._debug = {}

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        c = imgs.shape[1]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        
        # Optimization: Use permute + reshape instead of einsum
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = x.permute(0, 2, 4, 3, 5, 1) # [N, h, w, p, p, c]
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def _sector_mask_to_patch_valid(self, sector_mask, L: int, device: torch.device, return_ratio: bool = False):
        if sector_mask is None:
            pv = torch.ones((1, L), dtype=torch.bool, device=device)
            vr = torch.ones((1, L), dtype=torch.float32, device=device)
            return (pv, vr) if return_ratio else pv

        if sector_mask.dim() == 3:
            sector_mask = sector_mask.unsqueeze(1)
        if sector_mask.dim() != 4:
            raise ValueError(f"sector_mask must be [N,1,H,W] or [N,H,W], got {tuple(sector_mask.shape)}")

        p = self.patch_embed.patch_size[0]
        m = sector_mask.to(device=device, dtype=torch.float32)
        pooled = F.avg_pool2d(m, kernel_size=p, stride=p)
        valid_ratio = pooled.flatten(1)

        tau = self.roi_tau
        patch_valid = (pooled.flatten(1) >= tau)

        bad = patch_valid.sum(dim=1) <= 0
        if bad.any():
            patch_valid[bad] = True
            valid_ratio[bad] = 1.0

        if patch_valid.shape[1] != L:
            raise ValueError(f"patch_valid length mismatch: got {patch_valid.shape[1]}, expected {L}")
        return (patch_valid, valid_ratio) if return_ratio else patch_valid

    def grid_patchify(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        return x

    def _sanitize_p_polar(self, p_polar: torch.Tensor | None, N: int, L: int, device: torch.device) -> torch.Tensor | None:
        """
        期望 p_polar 为 [N,L] 概率分布（>=0，行和=1）
        dataset 侧可能输出 float32，可能在 CPU，这里统一搬到 device 并做安全归一化
        """
        if p_polar is None:
            return None
        p_polar = p_polar.to(device=device, dtype=torch.float32)
        if p_polar.dim() != 2 or p_polar.shape[0] != N or p_polar.shape[1] != L:
            raise ValueError(f"p_polar must be [N,L]=[{N},{L}], got {tuple(p_polar.shape)}")
        p_polar = torch.nan_to_num(p_polar, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        p_polar = p_polar / (p_polar.sum(dim=1, keepdim=True) + 1e-6)
        return p_polar

    def random_masking(self, x, mask_ratio, kept_mask_ratio, imgs, sel_idx=None, p_polar=None):
        """
        新版：只在 dataset 提供的 sel_idx=[N,K] 子集内做 selection/masking
        - sel_idx 必须来自 ROI 内（dataset 已保证）
        - batch 内 K 必须相同（sampler+collate 已保证）
        """
        N, L, D = x.shape
        device = x.device

        if sel_idx is None:
            raise ValueError("sel_idx is required in bucketed training mode (dataset already provides it).")
        sel_idx = sel_idx.to(device=device, dtype=torch.long)
        if sel_idx.dim() != 2 or sel_idx.shape[0] != N:
            raise ValueError(f"sel_idx must be [N,K], got {tuple(sel_idx.shape)}")

        K = int(sel_idx.shape[1])
        if K > L:
            sel_idx = sel_idx[:, :L]
            K = L
        K = max(1, K)

        # ---------------- HOG prob on full grid ----------------
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                Gh = int(self.patch_embed.grid_size[0])
                Gw = int(self.patch_embed.grid_size[1])
                logits = self.hog(imgs.float(), Gh, Gw)  # [N,L]
        logits = torch.nan_to_num(logits)
        p_hog = self.softmax(logits)  # [N,L]

        # ---------------- MIX with p_polar (from dataset) ----------------
        p_polar = self._sanitize_p_polar(p_polar, N=N, L=L, device=device)

        if p_polar is None:
            p_x = p_hog
        else:
            lam = self.lam_polar
            p_mix = (1.0 - lam) * p_hog + lam * p_polar
            p_mix = p_mix / (p_mix.sum(dim=1, keepdim=True) + 1e-6)
            p_x = p_mix

        # ---------------- restrict to sel_idx subset ----------------
        # x_sub: [N,K,D], p_sub: [N,K]
        #x_sub = torch.gather(x, 1, sel_idx.unsqueeze(-1).expand(-1, -1, D))
        p_sub = torch.gather(p_x, 1, sel_idx)  # [N,K]
        p_sub = torch.nan_to_num(p_sub, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        row = p_sub.sum(dim=1, keepdim=True)
        bad = row.squeeze(1) <= 0
        if bad.any():
            p_sub[bad] = 1.0
            row = p_sub.sum(dim=1, keepdim=True)
        p_sub = p_sub / (row + 1e-6)

        # ---------------- choose keep/selected sizes based on K ----------------
        min_keep = 2
        len_keep = int(round(K * (1.0 - float(mask_ratio))))
        len_keep = max(min_keep, min(len_keep, K))

        len_selected_initialization = max(1, min(len_keep // 2, K - 1))

        # sample selected positions within K
        sel_pos = torch.multinomial(p_sub, num_samples=len_selected_initialization, replacement=False)  # [N,Spos]
        ids_selected = torch.gather(sel_idx, 1, sel_pos)  # map back to original [0..L-1] indices

        if (K - len_selected_initialization) <= 0:
            masked_ids_padded = sel_idx[:, :1].clone()
            masked_valid_mask = torch.ones((N, 1), dtype=torch.bool, device=device)
            mask = torch.zeros((N, L), dtype=torch.bool, device=device)
            mask.scatter_(1, masked_ids_padded, True)
            select_token = torch.gather(x, 1, ids_selected.unsqueeze(-1).expand(-1, -1, D))
            return select_token, mask, N, L, p_x, masked_ids_padded, masked_valid_mask

        # unselected = remaining positions in sel_idx
        selected_mask = torch.zeros((N, K), dtype=torch.bool, device=device)
        selected_mask.scatter_(1, sel_pos, True)
        # ids_unselected are original indices too
        ids_unselected = sel_idx[~selected_mask].view(N, K - len_selected_initialization)

        # gather tokens from original x using original indices
        select_token = torch.gather(x, 1, ids_selected.unsqueeze(-1).expand(-1, -1, D))
        unselect_token = torch.gather(x, 1, ids_unselected.unsqueeze(-1).expand(-1, -1, D))

        # token expansion（保持你原逻辑）
        (select_token, ids_selected), (unselect_token, ids_unselected) = self.token_select.token_expansion(
            select_token, ids_selected, unselect_token, ids_unselected, x
        )

        # ---------------- mask sampling strictly within ids_unselected ----------------
        p_un = torch.gather(p_x, 1, ids_unselected)  # [N,U]
        U = int(p_un.shape[1])

        if U <= 0:
            masked_ids_padded = sel_idx[:, :1].clone()
            masked_valid_mask = torch.ones((N, 1), dtype=torch.bool, device=device)
            mask = torch.zeros((N, L), dtype=torch.bool, device=device)
            mask.scatter_(1, masked_ids_padded, True)
            return select_token, mask, N, L, p_x, masked_ids_padded, masked_valid_mask

        p_un = torch.nan_to_num(p_un, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        row_un = p_un.sum(dim=1, keepdim=True)
        bad_un = row_un.squeeze(1) <= 0
        if bad_un.any():
            p_un[bad_un] = 1.0
            row_un = p_un.sum(dim=1, keepdim=True)
        p_un = p_un / (row_un + 1e-6)

        # 注意：现在 K 是 batch 固定的，所以 Mi 也可统一
        Mi = int(round(K * float(kept_mask_ratio)))
        Mi = max(1, min(Mi, U))
        if self.mask_cap is not None:
            Mi = min(Mi, int(self.mask_cap))

        Mmax = Mi  # batch 内固定
        Mmax = max(1, min(Mmax, U))

        new_idx_full = torch.multinomial(p_un, num_samples=Mmax, replacement=False)  # [N,Mmax]
        masked_ids_padded = torch.gather(ids_unselected, 1, new_idx_full)            # [N,Mmax] (original indices)
        masked_valid_mask = torch.ones((N, Mmax), dtype=torch.bool, device=device)   # 全有效（不再 padding）

        # build mask [N,L]
        mask = torch.zeros((N, L), dtype=torch.bool, device=device)
        batch_ids = torch.arange(N, device=device).unsqueeze(1).expand(-1, Mmax)
        mask[batch_ids.reshape(-1), masked_ids_padded.reshape(-1)] = True

        with torch.no_grad():
            self._debug = {
                "K": torch.tensor(float(K), device=device),
                "S": torch.tensor(float(select_token.shape[1]), device=device),
                "Mi": torch.tensor(float(Mi), device=device),
                "Mmax": torch.tensor(float(Mmax), device=device),
                "masked_total": mask.sum().float(),
            }

        return select_token, mask, N, L, p_x, masked_ids_padded, masked_valid_mask

    def forward_encoder(self, x, mask_ratio, kept_mask_ratio, sel_idx=None, p_polar=None):
        imgs = x
        x = self.grid_patchify(x)

        x, mask, N, L, p_x, masked_ids_padded, masked_valid_mask = self.random_masking(
            x, mask_ratio, kept_mask_ratio, imgs,
            sel_idx=sel_idx,
            p_polar=p_polar,
        )

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x, mask, N, L, None, p_x, masked_ids_padded, masked_valid_mask


    def mask_tokens_grid(self, masked_ids_padded, masked_valid_mask):
        N, Mmax = masked_ids_padded.shape
        pos = self.decoder_pos_embed[:, 1:, :].expand(N, -1, -1)
        x = torch.gather(pos, 1, masked_ids_padded.unsqueeze(-1).expand(-1, -1, pos.shape[-1]))
        x = x + self.mask_token
        x = x * masked_valid_mask.unsqueeze(-1).to(x.dtype)
        return x

    def forward_decoder(self, y, masked_ids_padded, masked_valid_mask):
        y = self.decoder_embed1(y)
        y = y + self.decoder_pos_embed[:, : y.shape[1], :]
        x = self.mask_tokens_grid(masked_ids_padded, masked_valid_mask)
        for blk in self.decoder_blocks:
            x = blk(x, y)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward_loss_sampling(self, imgs, pred, masked_ids_padded, masked_valid_mask):
        target = self.patchify(imgs)
        _, _, P = target.shape
        target_pad = torch.gather(target, 1, masked_ids_padded.unsqueeze(-1).expand(-1, -1, P))

        if self.norm_pix_loss:
            mean = target_pad.mean(dim=-1, keepdim=True)
            var = target_pad.var(dim=-1, keepdim=True)
            target_pad = (target_pad - mean) / (var + 1e-6) ** 0.5

        loss_func = nn.MSELoss(reduction="none")
        loss_main = torch.mean(loss_func(pred, target_pad), dim=-1)

        v = masked_valid_mask.float()
        denom = v.sum(dim=1).clamp_min(1.0)
        loss = (loss_main * v).sum(dim=1) / denom
        return loss

    def forward(self, imgs, sel_idx=None, p_polar=None, mask_ratio=0.75, kept_mask_ratio=0.5):
        with torch.cuda.amp.autocast(enabled=imgs.is_cuda):
            latent, mask, N, L, coords, p_x, masked_ids_padded, masked_valid_mask = self.forward_encoder(
                imgs, mask_ratio, kept_mask_ratio, sel_idx=sel_idx, p_polar=p_polar
            )
            pred = self.forward_decoder(latent, masked_ids_padded, masked_valid_mask)

        loss_per_sample = self.forward_loss_sampling(imgs.float(), pred.float(), masked_ids_padded, masked_valid_mask)
        return loss_per_sample



# ---------- model factories ----------
def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_huge_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch16 = mae_vit_huge_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
