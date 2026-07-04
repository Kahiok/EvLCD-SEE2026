"""
EvLCD: Event-Guided WDR Image Restoration
         with Local Color Distribution Embedded (LCDE) Module
         and Brightness-Prompt Conditioning

Architecture overview:
  - Multi-scale event encoder (64-ch voxel grid → 48-ch features at 3 scales)
  - LCDE module: LCD pyramid + guided mask + DualIllumNet (adapted from LCDPNet)
  - U-Net backbone (IGAB blocks + FiLM conditioning with brightness prompt)
  - MDTA (Multi-Dconv Head Transposed Attention) refinement in decoder
  - 15.85 M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB

NL_COND_DIM = 256


# ── Attention blocks ───────────────────────────────────────────────────────────

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class IGMSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x):
        b, h, w, c = x.shape
        x_flat = x.reshape(b, h * w, c)
        q = rearrange(self.to_q(x_flat), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.to_k(x_flat), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.to_v(x_flat), "b n (h d) -> b h n d", h=self.heads)
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 3, 1, 2).reshape(b, h * w, self.heads * self.dim_head)
        out_c = self.proj(out).view(b, h, w, c)
        out_p = (
            self.pos_emb(v.reshape(b, h, w, c).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
        )
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                IGMSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ])
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


# ── Residual / ECA block ───────────────────────────────────────────────────────

class ECAResidualBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.norm = nn.InstanceNorm2d(nf // 2, affine=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out1, out2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out1), out2], dim=1)
        out = self.relu(out)
        out = self.conv2(out)
        y = self.avg_pool(out).squeeze(-1).squeeze(-1)
        y = self.eca_conv(y.unsqueeze(1))
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        out = out * y.expand_as(out)
        return self.relu(out + res)


# ── LCDE components (adapted from LCDPNet) ─────────────────────────────────────

class LCDPyramid(nn.Module):
    def __init__(self, scales=(4, 8, 16), out_ch=32):
        super().__init__()
        self.scales = scales
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(6, out_ch, 1, bias=False), nn.ReLU(inplace=True))
            for _ in scales
        ])
        self.aggregator = nn.Sequential(
            nn.Conv2d(out_ch * len(scales), out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feats = []
        for scale, proj in zip(self.scales, self.projections):
            pad = scale // 2
            mean = F.avg_pool2d(x, kernel_size=scale, stride=1, padding=pad)[:, :, :H, :W]
            sq_mean = F.avg_pool2d(x ** 2, kernel_size=scale, stride=1, padding=pad)[:, :, :H, :W]
            std = torch.sqrt((sq_mean - mean ** 2).clamp(min=0) + 1e-8)
            feats.append(proj(torch.cat([mean, std], dim=1)))
        return self.aggregator(torch.cat(feats, dim=1))


class LCDGuidedMask(nn.Module):
    def __init__(self, lcd_ch=32, rgb_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(lcd_ch + rgb_ch, 64, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, bias=True),
        )

    def forward(self, rgb, lcd_feat):
        probs = F.softmax(self.net(torch.cat([rgb, lcd_feat], dim=1)), dim=1)
        return probs[:, 0:1], probs[:, 1:2], probs[:, 2:3]


class DualIllumNet(nn.Module):
    def __init__(self, base_ch=24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
        )
        self.under_dec = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1, bias=True), nn.ReLU(inplace=True),
        )
        self.over_dec = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1, bias=True), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat = self.encoder(x)
        under_enhanced = (x * self.under_dec(feat) + x).clamp(0, 1)
        x_rev = 1.0 - x
        L_prime = self.over_dec(self.encoder(x_rev))
        over_enhanced = (1.0 - (x_rev * L_prime + x_rev)).clamp(0, 1)
        return under_enhanced, over_enhanced, None, None


# ── Event encoder & structure branch ──────────────────────────────────────────

class TemporalMixer(nn.Module):
    def __init__(self, T=64):
        super().__init__()
        self.conv_k3 = nn.Conv2d(1, 4, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_k5 = nn.Conv2d(1, 4, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.proj    = nn.Conv2d(8, 1, kernel_size=(1, 1), bias=False)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        B, T, H, W = x.shape
        x_2d = x.reshape(B, 1, T, H * W)
        c3 = self.relu(self.conv_k3(x_2d))
        c5 = self.relu(self.conv_k5(x_2d))
        delta = self.proj(torch.cat([c3, c5], dim=1))
        delta = delta.reshape(B, T, H, W)
        return x + delta


class MultiScaleEventEncoder(nn.Module):
    def __init__(self, event_ch=64, base_ch=48):
        super().__init__()
        self.scale_0 = nn.Sequential(
            nn.Conv2d(event_ch, base_ch, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            ECAResidualBlock(base_ch),
        )
        self.scale_1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ECAResidualBlock(base_ch),
        )
        self.scale_2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ECAResidualBlock(base_ch),
        )

    def forward(self, x):
        ev_h  = self.scale_0(x)
        ev_h2 = self.scale_1(ev_h)
        ev_h4 = self.scale_2(ev_h2)
        return ev_h, ev_h2, ev_h4


class StructureBranch(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        self.edge_struct_h  = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU())
        self.edge_struct_h2 = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU())
        self.edge_struct_h4 = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU())
        self.edge_head_h  = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=True), nn.Sigmoid())
        self.edge_head_h2 = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=True), nn.Sigmoid())
        self.edge_head_h4 = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=True), nn.Sigmoid())
        self.sf_h  = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU())
        self.sf_h2 = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU())
        self.sf_h4 = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU())

    def forward(self, ev_h, ev_h2, ev_h4, training=True):
        struct_feats = (self.sf_h(ev_h), self.sf_h2(ev_h2), self.sf_h4(ev_h4))
        edge_preds = None
        if training:
            edge_preds = [
                self.edge_head_h(self.edge_struct_h(ev_h)),
                self.edge_head_h2(self.edge_struct_h2(ev_h2)),
                self.edge_head_h4(self.edge_struct_h4(ev_h4)),
            ]
        return struct_feats, edge_preds


class LCD_Enhance(nn.Module):
    def __init__(self, channel, depth=2):
        super().__init__()
        self.img_blocks = nn.ModuleList([ECAResidualBlock(channel) for _ in range(depth)])
        self.ev_blocks  = nn.ModuleList([ECAResidualBlock(channel) for _ in range(depth)])
        self.ev_proj = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, img_feat, ev_feat, problem_prob):
        H, W = img_feat.shape[-2:]
        prob = problem_prob
        if prob.shape[-2:] != (H, W):
            prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=False)
        for img_blk, ev_blk in zip(self.img_blocks, self.ev_blocks):
            img_feat = img_blk(img_feat)
            ev_feat  = ev_blk(ev_feat)
        return img_feat + prob * self.ev_proj(ev_feat)


# ── Brightness prompt & FiLM conditioning ─────────────────────────────────────

class BrightnessPromptMLP(nn.Module):
    """Maps scalar mean brightness B ∈ [0,1] → 256-dim FiLM conditioning vector."""
    def __init__(self, hidden=128, out_dim=NL_COND_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: x = x * (1 + γ) + β, near-zero init."""
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        out = self.proj(cond)
        gamma, beta = out.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


# ── U-Net backbone ─────────────────────────────────────────────────────────────

INJECT_SCALE = 0.1


class Unet_ReFormer_LCD(nn.Module):
    def __init__(self, dim=48, level=2, num_blocks=(2, 4, 4),
                 struct_ch=48, enhance_depth=(2, 4, 6),
                 nl_cond_dim=NL_COND_DIM):
        super().__init__()
        self.level = level
        self.img_head = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=False)
        self.ev_img_align = nn.Conv2d(dim * 2, dim, 1, bias=False)

        self.encoder_layers = nn.ModuleList()
        self.enc_inject = nn.ModuleList()
        dim_cur = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_cur, num_blocks=num_blocks[i], dim_head=dim, heads=dim_cur // dim),
                nn.Conv2d(dim_cur, dim_cur * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_cur, dim_cur * 2, 4, 2, 1, bias=False),
                LCD_Enhance(dim_cur, depth=enhance_depth[i]),
            ]))
            self.enc_inject.append(nn.Conv2d(struct_ch, dim_cur, 1, bias=True))
            dim_cur *= 2

        self.bottleneck_lcd = LCD_Enhance(dim_cur, depth=enhance_depth[-1])
        self.bot_inject = nn.Conv2d(struct_ch, dim_cur, 1, bias=True)
        self.bottleneck = IGAB(dim=dim_cur, dim_head=dim, heads=dim_cur // dim,
                               num_blocks=num_blocks[-1])
        self.film_bot = FiLMBlock(nl_cond_dim, dim_cur)

        self.decoder_layers = nn.ModuleList()
        self.dec_inject = nn.ModuleList()
        self.film_dec = nn.ModuleList()
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_cur, dim_cur // 2, 2, 2),
                nn.Conv2d(dim_cur, dim_cur // 2, 1, bias=False),
                LCD_Enhance(dim_cur // 2, depth=enhance_depth[level - 1 - i]),
                IGAB(dim=dim_cur // 2, dim_head=dim, heads=(dim_cur // 2) // dim,
                     num_blocks=num_blocks[level - 1 - i]),
            ]))
            self.dec_inject.append(nn.Conv2d(struct_ch, dim_cur // 2, 1, bias=True))
            self.film_dec.append(FiLMBlock(nl_cond_dim, dim_cur // 2))
            dim_cur //= 2

        self.mapping = nn.Conv2d(dim, 3, 3, 1, 1, bias=False)

    def _inject(self, fused, proj_layer, struct_feat):
        H, W = fused.shape[-2:]
        sf = struct_feat
        if sf.shape[-2:] != (H, W):
            sf = F.interpolate(sf, (H, W), mode='bilinear', align_corners=False)
        return fused + INJECT_SCALE * proj_layer(sf)

    def forward(self, rgb, ev_feat, problem_prob, struct_feats, nl_cond):
        sf_h, sf_h2, sf_h4 = struct_feats
        struct_enc = [sf_h, sf_h2]
        struct_dec = [sf_h2, sf_h]

        img_feat = self.lrelu(self.img_head(rgb))
        fused = self.ev_img_align(torch.cat([img_feat, ev_feat], dim=1))

        enc_skips, ev_skips = [], [ev_feat]
        ev_cur = ev_feat
        for i, (IGAB_, FeaDown, EvDown, LCD_enh) in enumerate(self.encoder_layers):
            fused = IGAB_(fused)
            fused = LCD_enh(fused, ev_cur, problem_prob)
            fused = self._inject(fused, self.enc_inject[i], struct_enc[i])
            enc_skips.append(fused)
            fused = FeaDown(fused)
            ev_cur = EvDown(ev_cur)
            ev_skips.append(ev_cur)

        fused = self.bottleneck_lcd(fused, ev_cur, problem_prob)
        fused = self._inject(fused, self.bot_inject, sf_h4)
        fused = self.bottleneck(fused)
        fused = self.film_bot(fused, nl_cond)

        for i, (FeaUp, Fusion, LCD_enh, IGAB_) in enumerate(self.decoder_layers):
            fused = FeaUp(fused)
            skip = enc_skips[self.level - 1 - i]
            ev_skip = ev_skips[self.level - 1 - i]
            fused = Fusion(torch.cat([fused, skip], dim=1))
            fused = LCD_enh(fused, ev_skip, problem_prob)
            fused = self._inject(fused, self.dec_inject[i], struct_dec[i])
            fused = IGAB_(fused)
            fused = self.film_dec[i](fused, nl_cond)

        return self.mapping(fused)


# ── MDTA block (Restormer-style channel-wise transposed attention) ─────────────

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias   = nn.Parameter(torch.zeros(c))
        self.eps    = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MDTABlock(nn.Module):
    """Single Restormer-style MDTA block; zero-init output → identity at cold start."""

    def __init__(self, c, num_heads=1, ffn_expand=2, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = LayerNorm2d(c)
        self.q_pw  = nn.Conv2d(c, c, 1, bias=bias)
        self.k_pw  = nn.Conv2d(c, c, 1, bias=bias)
        self.v_pw  = nn.Conv2d(c, c, 1, bias=bias)
        self.q_dw  = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=bias)
        self.k_dw  = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=bias)
        self.v_dw  = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=bias)
        self.proj  = nn.Conv2d(c, c, 1, bias=bias)

        ffn_ch = int(c * ffn_expand)
        self.norm2    = LayerNorm2d(c)
        self.ffn_pw1  = nn.Conv2d(c, ffn_ch, 1, bias=bias)
        self.ffn_dw   = nn.Conv2d(ffn_ch, ffn_ch, 3, padding=1, groups=ffn_ch, bias=bias)
        self.ffn_pw2  = nn.Conv2d(ffn_ch, c, 1, bias=bias)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.ffn_pw2.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.num_heads
        d = C // h

        res = x
        x_n = self.norm1(x)
        q = self.q_dw(self.q_pw(x_n))
        k = self.k_dw(self.k_pw(x_n))
        v = self.v_dw(self.v_pw(x_n))

        q = q.reshape(B, h, d, H * W)
        k = k.reshape(B, h, d, H * W)
        v = v.reshape(B, h, d, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, H, W)
        out = self.proj(out) + res

        res = out
        out = self.norm2(out)
        out = F.gelu(self.ffn_dw(self.ffn_pw1(out)))
        out = self.ffn_pw2(out) + res

        return out


# ── U-Net with MDTA appended after last decoder layer ─────────────────────────

class Unet_ReFormer_LCD_MDTA(Unet_ReFormer_LCD):
    def __init__(self, dim=48, level=2, num_blocks=(2, 4, 4),
                 struct_ch=48, enhance_depth=(2, 4, 6), nl_cond_dim=NL_COND_DIM):
        super().__init__(dim=dim, level=level, num_blocks=num_blocks,
                         struct_ch=struct_ch, enhance_depth=enhance_depth,
                         nl_cond_dim=nl_cond_dim)
        self.mdta_out = MDTABlock(c=dim, num_heads=1, ffn_expand=2)

    def forward(self, rgb, ev_feat, problem_prob, struct_feats, nl_cond):
        sf_h, sf_h2, sf_h4 = struct_feats
        struct_enc = [sf_h, sf_h2]
        struct_dec = [sf_h2, sf_h]

        img_feat = self.lrelu(self.img_head(rgb))
        fused = self.ev_img_align(torch.cat([img_feat, ev_feat], dim=1))

        enc_skips, ev_skips = [], [ev_feat]
        ev_cur = ev_feat
        for i, (IGAB_, FeaDown, EvDown, LCD_enh) in enumerate(self.encoder_layers):
            fused = IGAB_(fused)
            fused = LCD_enh(fused, ev_cur, problem_prob)
            fused = self._inject(fused, self.enc_inject[i], struct_enc[i])
            enc_skips.append(fused)
            fused = FeaDown(fused)
            ev_cur = EvDown(ev_cur)
            ev_skips.append(ev_cur)

        fused = self.bottleneck_lcd(fused, ev_cur, problem_prob)
        fused = self._inject(fused, self.bot_inject, sf_h4)
        fused = self.bottleneck(fused)
        fused = self.film_bot(fused, nl_cond)

        for i, (FeaUp, Fusion, LCD_enh, IGAB_) in enumerate(self.decoder_layers):
            fused = FeaUp(fused)
            skip = enc_skips[self.level - 1 - i]
            ev_skip = ev_skips[self.level - 1 - i]
            fused = Fusion(torch.cat([fused, skip], dim=1))
            fused = LCD_enh(fused, ev_skip, problem_prob)
            fused = self._inject(fused, self.dec_inject[i], struct_dec[i])
            fused = IGAB_(fused)
            fused = self.film_dec[i](fused, nl_cond)

        fused = self.mdta_out(fused)
        return self.mapping(fused)


# ── EvLCD main model ───────────────────────────────────────────────────────────

class _EvLCDBase(nn.Module):
    """Base model: v12 backbone with BrightnessPromptMLP conditioning."""

    def __init__(self, event_ch=64, base_ch=48, lcd_ch=32):
        super().__init__()
        self.lcd_pyramid    = LCDPyramid(scales=(4, 8, 16), out_ch=lcd_ch)
        self.lcd_mask       = LCDGuidedMask(lcd_ch=lcd_ch, rgb_ch=3)
        self.dual_illum     = DualIllumNet(base_ch=base_ch // 2)
        self.temporal_mixer = TemporalMixer(T=event_ch)
        self.ev_encoder     = MultiScaleEventEncoder(event_ch=event_ch, base_ch=base_ch)
        self.struct_branch  = StructureBranch(ch=base_ch)
        self.nl_stats_mlp   = BrightnessPromptMLP(hidden=128, out_dim=NL_COND_DIM)
        self.unet = Unet_ReFormer_LCD(
            dim=base_ch, level=2, num_blocks=(2, 4, 4),
            struct_ch=base_ch, enhance_depth=(2, 4, 6),
            nl_cond_dim=NL_COND_DIM,
        )

    def forward(self, batch):
        rgb    = batch[ELB.LL].clamp(0, 1)
        events = batch[ELB.E]
        B, _, H, W = rgb.shape

        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h > 0 or pad_w > 0:
            rgb    = F.pad(rgb,    (0, pad_w, 0, pad_h), mode='reflect')
            events = F.pad(events, (0, pad_w, 0, pad_h), mode='reflect')

        nl_stats = batch.get("NL_VID_STATS")
        if nl_stats is not None and isinstance(nl_stats, torch.Tensor):
            nl_cond = self.nl_stats_mlp(nl_stats.to(rgb.device))
        else:
            nl_cond = torch.zeros(B, NL_COND_DIM, device=rgb.device)

        lcd_feat = self.lcd_pyramid(rgb)
        over_prob, under_prob, normal_prob = self.lcd_mask(rgb, lcd_feat)
        problem_prob = over_prob + under_prob

        under_enh, over_enh, _, _ = self.dual_illum(rgb)

        events_mixed = self.temporal_mixer(events)
        ev_h, ev_h2, ev_h4 = self.ev_encoder(events_mixed)

        struct_feats, edge_preds = self.struct_branch(
            ev_h, ev_h2, ev_h4, training=self.training
        )

        if self.training and edge_preds is not None:
            batch['edge_pred_list'] = edge_preds
            batch['problem_mask'] = problem_prob.detach()

        unet_out = self.unet(rgb, ev_h, problem_prob, struct_feats, nl_cond)

        base  = under_enh * under_prob + over_enh * over_prob + rgb * normal_prob
        final = (unet_out + base).clamp(0, 1)

        if pad_h > 0 or pad_w > 0:
            final = final[:, :, :H, :W]

        batch[ELB.PRD] = final
        return batch


class EvLCD(_EvLCDBase):
    """
    EvLCD: Event-guided WDR restoration with LCDE module,
    brightness-prompt conditioning, and MDTA decoder refinement.
    """

    def __init__(self, event_ch=64, base_ch=48, lcd_ch=32, no_bprompt=False):
        super().__init__(event_ch=event_ch, base_ch=base_ch, lcd_ch=lcd_ch)
        self.no_bprompt = no_bprompt
        self.unet = Unet_ReFormer_LCD_MDTA(
            dim=base_ch, level=2, num_blocks=(2, 4, 4),
            struct_ch=base_ch, enhance_depth=(2, 4, 6),
            nl_cond_dim=NL_COND_DIM,
        )
        n      = sum(p.numel() for p in self.parameters())
        n_mdta = sum(p.numel() for p in self.unet.mdta_out.parameters())
        print(f"[EvLCD] total params: {n:,}  (MDTABlock: {n_mdta:,})")

    def forward(self, batch):
        if self.no_bprompt:
            batch["NL_VID_STATS"] = None
        elif "MANIFEST_BRIGHTNESS" in batch and isinstance(batch["MANIFEST_BRIGHTNESS"], torch.Tensor):
            b = batch["MANIFEST_BRIGHTNESS"].float()
            if b.dim() == 1:
                b = b.unsqueeze(1)
            batch["NL_VID_STATS"] = b
        else:
            nl = batch.get(ELB.NL)
            if nl is not None and isinstance(nl, torch.Tensor) and nl.dim() == 4:
                b_prompt = nl.mean(dim=[1, 2, 3]).unsqueeze(1)
                batch["NL_VID_STATS"] = b_prompt
        return _EvLCDBase.forward(self, batch)


LCDEvLight = EvLCD  # backward-compatible alias for old configs
