"""Noise-prediction training and deterministic reverse sampling from the old CNNDM."""
import torch
from torch import nn
import torch.nn.functional as F

from .channel import response
from .unet import UNet


class Diffusion(nn.Module):
    def __init__(self, target, config):
        super().__init__()
        self.target = target
        self.config = dict(config)
        self.n_T = config['steps']
        channels = 2 if target == 'y' else 1
        self.eps_model = UNet(
            in_channel=channels, out_channel=channels,
            inner_channel=config['inner_channels'], norm_groups=config['norm_groups'],
            channel_mults=tuple(config['channel_mults']), attn_res=config['attention_resolutions'],
            res_blocks=config['res_blocks'], image_size=config['side'], version='v2')
        beta = torch.cat((torch.zeros(1), torch.linspace(config['beta_start'], config['beta_end'], self.n_T)))
        alpha = 1 - beta
        alphabar = alpha.cumprod(0)
        # Names retained for explicit loading of legacy noise-predictor weights.
        for name, value in {
            'alpha_t': alpha, 'oneover_sqrta': alpha.rsqrt(), 'sqrt_beta_t': beta.sqrt(),
            'alphabar_t': alphabar, 'sqrtab': alphabar.sqrt(),
            'sqrtmab': (1 - alphabar).sqrt(),
            'mab_over_sqrtmab': beta / (1 - alphabar).sqrt().clamp_min(1e-12),
        }.items():
            self.register_buffer(name, value)

    def forward(self, clean):
        t = torch.randint(1, self.n_T + 1, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.sqrtab[t, None, None, None] * clean + self.sqrtmab[t, None, None, None] * noise
        return F.mse_loss(self.eps_model(noisy, t.float() / self.n_T), noise)

    @torch.no_grad()
    def denoise(self, noisy, variance):
        variance = torch.as_tensor(variance, device=noisy.device)
        if variance.numel() != 1 or variance < 0 or not torch.isfinite(variance):
            raise ValueError('Diffusion variance must be a finite nonnegative scalar')
        start = int((variance - (1 - self.alphabar_t) / self.alphabar_t).abs().argmin())
        if start == 0:
            return noisy
        # y = x0 + sigma*eps must be scaled into the forward diffusion distribution.
        x = noisy * self.sqrtab[start]
        for step in range(start, 0, -1):
            # Training and sampling use the same T, not the SNR-dependent start index.
            t = torch.full((len(x),), step / self.n_T, device=x.device)
            noise = self.eps_model(x, t)
            clean = (x - self.sqrtmab[step] * noise) / self.sqrtab[step]
            x = self.sqrtab[step - 1] * clean + self.sqrtmab[step - 1] * noise
        return x


def synthetic_clean(target, batch, device, config):
    """Legacy synthetic distributions, generated online without image data."""
    side = config['diffusion']['side']
    h = response(batch, side * side, device, config['channel']).reshape(batch, 1, side, side)
    if target == 'h':
        # Train on both components used at inference, with one shared predictor.
        return torch.cat((h.real, h.imag), dim=0)
    x = torch.randn(batch, 1, side, side, device=device)
    if target == 'x':
        return x
    hx = h * x
    return torch.cat((hx.real, hx.imag), dim=1)
