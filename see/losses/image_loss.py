import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


# ── DoG edge utilities ────────────────────────────────────────────────

def _gaussian_kernel(size, sigma, device):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)  # size × size


def dog_edge(img, sigma1=1.0, sigma2=2.0):
    """
    Difference-of-Gaussians edge map from RGB image.
    More robust than Sobel for HDR restoration: avoids texture/noise activation.
    Returns (B, 1, H, W) normalised to [0, 1] per image.
    """
    def _blur(t, sigma):
        k = int(6 * sigma) + 1
        if k % 2 == 0:
            k += 1
        kern = _gaussian_kernel(k, sigma, t.device).unsqueeze(0).unsqueeze(0)
        B, C, H, W = t.shape
        flat = t.reshape(B * C, 1, H, W)
        out = F.conv2d(flat, kern, padding=k // 2)
        return out.reshape(B, C, H, W)

    g1 = _blur(img, sigma1)
    g2 = _blur(img, sigma2)
    dog = (g1 - g2).abs().mean(dim=1, keepdim=True)           # B, 1, H, W
    dog = dog / dog.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return dog


class EdgeAuxLoss(_Loss):
    """
    Multi-scale edge auxiliary loss for Event Encoder supervision.

    Reads batch['edge_pred_list']: list of (B,1,H_s,W_s) tensors.
    Computes DoG GT from batch[ELB.NL] at matching scales.
    Returns average L1 across scales.
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
        if 'edge_pred_list' not in batch:
            return batch[ELB.PRD].new_zeros(1).squeeze()

        gt = batch[ELB.NL]                    # B, 3, H, W
        gt_edge_full = dog_edge(gt)            # B, 1, H, W
        problem_mask = batch.get('problem_mask', None)  # B, 1, H, W or None

        total = gt.new_zeros(1).squeeze()
        for pred in batch['edge_pred_list']:
            th, tw = pred.shape[-2:]
            if gt_edge_full.shape[-2:] != (th, tw):
                gt_s = F.interpolate(gt_edge_full, size=(th, tw),
                                     mode='bilinear', align_corners=False)
            else:
                gt_s = gt_edge_full

            if problem_mask is not None:
                pm = problem_mask
                if pm.shape[-2:] != (th, tw):
                    pm = F.interpolate(pm, size=(th, tw),
                                       mode='bilinear', align_corners=False)
                total = total + (pm * (pred - gt_s).abs()).mean()
            else:
                total = total + F.l1_loss(pred, gt_s)

        return total / len(batch['edge_pred_list'])


class DistillationSupervision(_Loss):
    """
    Knowledge distillation loss for EXP-013.
    The Student model pre-computes MSE(f_student, f_teacher) during its
    forward pass and stores the scalar in batch['distill_loss'].
    This loss simply retrieves it (returns 0 during validation / when teacher absent).
    Registered under NAME='distillation_supervision' (suffix 'supervision' →
    EventLowLightBatchLoss calls self.loss(batch)).
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        loss_val = batch.get('distill_loss')
        if loss_val is None:
            from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
            return batch[ELB.PRD].new_zeros(1).squeeze()
        return loss_val


class LogBrightnessLoss(_Loss):
    """
    Log-domain per-image brightness alignment loss (EXP-011).

    L_bright = mean_over_batch( |log(mean(pred)+eps) - log(mean(GT)+eps)| )
             = mean of |alpha*|   where alpha* is the optimal ToneFit log-scale correction.

    Motivation (DIAG-006/008/010):
      - scale correction explains 91% of the +8 dB ToneFit upper bound
      - Corr(err_mean, PSNR) = -0.896
      - log domain aligns perfectly with alpha* (unlike linear |mean(pred)-mean(GT)|)
    """
    def __init__(self):
        super().__init__()

    def forward(self, gt, pred):
        pred_m = pred.mean(dim=[1, 2, 3])          # B
        gt_m   = gt.mean(dim=[1, 2, 3])            # B
        return (torch.log(pred_m + 1e-6) - torch.log(gt_m + 1e-6)).abs().mean()


def _batch_metadata_list(value, batch_size, default="unknown"):
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) < batch_size:
        values.extend([values[-1] if values else default] * (batch_size - len(values)))
    return values[:batch_size]


class ToneCurveAdjustmentLoss(_Loss):
    """
    Auxiliary supervision for the tone-curve head (v15 / EXP-016).

    Three sub-terms:
      1. global:  log(pred_mean / base_mean) should match log(gt_mean / base_mean)
      2. illum:   local_gain mean should match normal_illum / low_illum ratio
      3. alpha:   normal-normal samples should keep alpha < normal_alpha_target
    """
    def __init__(self, config):
        super().__init__()
        self.global_weight      = getattr(config, "global_weight",      1.0)
        self.illum_weight       = getattr(config, "illum_weight",       0.5)
        self.normal_alpha_weight = getattr(config, "normal_alpha_weight", 0.25)
        self.normal_alpha_target = getattr(config, "normal_alpha_target", 0.08)
        self.max_log_ratio      = getattr(config, "max_log_ratio",      0.916291)

    def forward(self, batch):
        from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB

        pred = batch.get(ELB.PRD)
        gt   = batch.get(ELB.NL)
        base = batch.get("base_pred_before_gain")
        if pred is None or gt is None or base is None:
            return batch[ELB.PRD].new_zeros(1).squeeze()

        eps = 1e-6
        base_detached = base.detach().clamp(0, 1)
        pred_mean = pred.mean(dim=(1, 2, 3)).clamp(min=eps)
        gt_mean   = gt.detach().mean(dim=(1, 2, 3)).clamp(min=eps)
        base_mean = base_detached.mean(dim=(1, 2, 3)).clamp(min=eps)

        pred_log_ratio   = torch.log(pred_mean / base_mean).clamp(-self.max_log_ratio, self.max_log_ratio)
        target_log_ratio = torch.log(gt_mean   / base_mean).clamp(-self.max_log_ratio, self.max_log_ratio)
        global_loss = F.smooth_l1_loss(pred_log_ratio, target_log_ratio)

        illum_loss = pred.new_zeros(1).squeeze()
        local_gain  = batch.get("pred_tone_local_gain")
        low_illum   = batch.get(ELB.ILL)
        normal_illum = batch.get(ELB.INL)
        if (
            isinstance(local_gain, torch.Tensor)
            and isinstance(low_illum, torch.Tensor)
            and isinstance(normal_illum, torch.Tensor)
        ):
            low_illum    = low_illum.to(device=pred.device, dtype=pred.dtype).clamp(min=eps)
            normal_illum = normal_illum.to(device=pred.device, dtype=pred.dtype).clamp(min=eps)
            if low_illum.shape[-2:] != local_gain.shape[-2:]:
                low_illum = F.interpolate(low_illum, local_gain.shape[-2:], mode="bilinear", align_corners=False)
            if normal_illum.shape[-2:] != local_gain.shape[-2:]:
                normal_illum = F.interpolate(normal_illum, local_gain.shape[-2:], mode="bilinear", align_corners=False)
            target_gain      = (normal_illum / low_illum).clamp(0.45, 2.25)
            pred_gain_mean   = local_gain.mean(dim=(1, 2, 3)).clamp(min=eps)
            target_gain_mean = target_gain.mean(dim=(1, 2, 3)).clamp(min=eps)
            illum_loss = F.smooth_l1_loss(torch.log(pred_gain_mean), torch.log(target_gain_mean))

        alpha_loss = pred.new_zeros(1).squeeze()
        alpha = batch.get("pred_tone_alpha")
        if isinstance(alpha, torch.Tensor):
            exposure_types = _batch_metadata_list(batch.get(ELB.LQET, "unknown"), pred.shape[0])
            normal_mask = alpha.new_tensor(
                [1.0 if str(et).lower().startswith("normal") else 0.0 for et in exposure_types]
            )
            if normal_mask.sum() > 0:
                alpha_mean  = alpha.mean(dim=(1, 2, 3))
                alpha_excess = (alpha_mean - self.normal_alpha_target).clamp(min=0)
                alpha_loss  = (alpha_excess * normal_mask).sum() / normal_mask.sum().clamp(min=1.0)

        return (
            self.global_weight      * global_loss
            + self.illum_weight     * illum_loss
            + self.normal_alpha_weight * alpha_loss
        )


class L1CharbonnierLoss(_Loss):
    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        diff_sq = diff * diff
        error = torch.sqrt(diff_sq + self.eps)
        loss = torch.mean(error)
        return loss


class GradientLoss(_Loss):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, i, j):
        b, c, h, w = i.shape
        idx = torch.abs(i[:, :, :, 1:] - i[:, :, :, : w - 1])
        idy = torch.abs(i[:, :, 1:, :] - i[:, :, : h - 1, :])
        jdx = torch.abs(j[:, :, :, 1:] - j[:, :, :, : w - 1])
        jdy = torch.abs(j[:, :, 1:, :] - j[:, :, : h - 1, :])
        loss = torch.mean(torch.abs(idx - jdx)) + torch.mean(torch.abs(idy - jdy))
        return loss


# Selfconstraints


class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()

        kernel_lf = torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
        kernel_rt = torch.tensor([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]]], dtype=torch.float32)
        kernel_up = torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
        kernel_dn = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]], dtype=torch.float32)

        self.weight_lf = nn.Parameter(kernel_lf, requires_grad=False)
        self.weight_rt = nn.Parameter(kernel_rt, requires_grad=False)
        self.weight_up = nn.Parameter(kernel_up, requires_grad=False)
        self.weight_dn = nn.Parameter(kernel_dn, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, original, enhanced):
        b, c, h, w = original.shape
        original_mean = torch.mean(original, dim=1, keepdim=True)
        enhanced_mean = torch.mean(enhanced, dim=1, keepdim=True)
        original_pool = self.pool(original_mean)
        enhanced_pool = self.pool(enhanced_mean)

        D_org_lf = F.conv2d(original_pool, self.weight_lf, padding=1)
        D_org_rt = F.conv2d(original_pool, self.weight_rt, padding=1)
        D_org_up = F.conv2d(original_pool, self.weight_up, padding=1)
        D_org_dn = F.conv2d(original_pool, self.weight_dn, padding=1)

        D_enh_lf = F.conv2d(enhanced_pool, self.weight_lf, padding=1)
        D_enh_rt = F.conv2d(enhanced_pool, self.weight_rt, padding=1)
        D_enh_up = F.conv2d(enhanced_pool, self.weight_up, padding=1)
        D_enh_dn = F.conv2d(enhanced_pool, self.weight_dn, padding=1)

        D_lf = torch.pow(D_org_lf - D_enh_lf, 2)
        D_rt = torch.pow(D_org_rt - D_enh_rt, 2)
        D_up = torch.pow(D_org_up - D_enh_up, 2)
        D_dn = torch.pow(D_org_dn - D_enh_dn, 2)

        E = D_lf + D_rt + D_up + D_dn
        E = E.mean()
        return E


# More Sample Constraints


class SEEMoreSampleConstraint(nn.Module):
    def __init__(self, config):
        super(SEEMoreSampleConstraint, self).__init__()
        self.config = config
        self.slw = config.spatial_loss_weight
        self.ecw = config.exposure_constancy_weight
        self.ccw = config.color_constancy_weight
        self.isw = config.ill_smooth_weight

        self.spatial_loss = SpatialConsistencyLoss()
        self.color_constancy = ColorConstancyRegularization()
        self.ill_smooth = IlluminationSmoothnessRegularization()

    def forward(self, nl, sc, ll, nlr, nlr_e):
        """
        nl: normal light. B 3 H W
        sc: self reconstructed. B 3 H W
        ll: low light. B 3 H W
        nlr: normal light reconstructed. B N 3 H W
        nlr_e: enhanced normal light reconstructed. B N
        """
        B, N, C, H, W = nlr.shape
        loss = 0
        # spatial loss
        loss = loss + self.spatial_loss(nl, sc) * self.slw
        for i in range(N):
            loss = loss + self.spatial_loss(nl, nlr[:, i, :, :, :]) * self.slw / N
        # exposure loss
        loss = loss + F.mse_loss(sc.mean([1, 2, 3], keepdim=True), ll.mean([1, 2, 3], keepdim=True)).mean() * self.ecw
        loss = loss + F.mse_loss(nlr.mean([2, 3, 4], keepdim=True), nlr_e).mean() * self.ecw
        # color constancy loss
        loss = loss + self.color_constancy(sc) * self.ccw
        for i in range(N):
            loss = loss + self.color_constancy(nlr[:, i, :, :, :]) * self.ccw / N
        # illumination smoothness loss
        loss = loss + self.ill_smooth(sc) * self.isw
        for i in range(N):
            loss = loss + self.ill_smooth(nlr[:, i, :, :, :]) * self.isw / N
        return loss


# Regularization Items


class ColorConstancyRegularization(_Loss):
    def __init__(self):
        super(ColorConstancyRegularization, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        k = torch.mean(k)
        return k


class ExposureControlRegularization(_Loss):
    def __init__(self, smoothing_kernal_size, expected_exposure_mean):
        super(ExposureControlRegularization, self).__init__()
        assert smoothing_kernal_size % 2 == 1
        assert expected_exposure_mean > 0.4 and expected_exposure_mean < 0.7
        self.pool = nn.AvgPool2d(smoothing_kernal_size)
        self.mean_val = torch.FloatTensor([expected_exposure_mean]).cuda()

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - self.mean_val, 2))
        return d


class IlluminationSmoothnessRegularization(_Loss):
    def __init__(self):
        super(IlluminationSmoothnessRegularization, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h - 1, :]), 2).mean()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w - 1]), 2).mean()
        loss = h_tv + w_tv
        return torch.mean(loss)


class HVILoss(_Loss):
    """
    L1 loss in HVI color space (Horizontal/Vertical/Intensity).

    HVI decouples brightness from color:
      Ĥ = Ck · S · cos(πH/3)
      V̂ = Ck · S · sin(πH/3)
      I  = max(R, G, B)
    Ck(x) = (sin(πx/2) + ε)^(1/k) collapses dark regions toward zero,
    suppressing noise in near-black pixels.  k is fixed here (not learnable).

    Reference: HVI-CIDNet+ (Yan et al., arXiv:2507.06814, 2025), Sec. III.
    """

    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k
        self.eps = 1e-8

    def _to_hvi(self, img: torch.Tensor) -> torch.Tensor:
        """Convert (B,3,H,W) sRGB in [0,1] → (B,3,H,W) HVI."""
        import math
        img = img.clamp(0.0, 1.0)
        R, G, B = img[:, 0], img[:, 1], img[:, 2]   # each (B,H,W)

        I_max = img.max(dim=1).values                 # (B,H,W)
        I_min = img.min(dim=1).values
        delta = (I_max - I_min).clamp(min=self.eps)
        S     = delta / I_max.clamp(min=self.eps)     # Saturation ∈ [0,1]

        # Hue in [0, 6) — additive masks keep gradients flowing
        mask_r = (I_max == R).float()
        mask_g = ((I_max == G) & (I_max != R)).float()
        mask_b = 1.0 - mask_r - mask_g

        # torch.remainder follows Python % convention: result sign matches divisor
        H_r = torch.remainder((G - B) / delta, 6.0)
        H_g = (B - R) / delta + 2.0
        H_b = (R - G) / delta + 4.0
        H   = mask_r * H_r + mask_g * H_g + mask_b * H_b   # (B,H,W)

        # Polarise hue → continuous at red boundary (H=0 and H=6 map to same point)
        angle = math.pi * H / 3.0    # [0, 2π]
        h_comp = torch.cos(angle)
        v_comp = torch.sin(angle)

        # Intensity collapse function (fixed k, no learnable param in loss)
        Ck = (torch.sin(math.pi * I_max / 2.0) + self.eps).pow(1.0 / self.k)

        H_hat = (Ck * S * h_comp).unsqueeze(1)   # (B,1,H,W)
        V_hat = (Ck * S * v_comp).unsqueeze(1)
        I_out = I_max.unsqueeze(1)
        return torch.cat([H_hat, V_hat, I_out], dim=1)   # (B,3,H,W)

    def forward(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            gt_hvi = self._to_hvi(gt)
        pred_hvi = self._to_hvi(pred)
        return F.l1_loss(pred_hvi, gt_hvi)
