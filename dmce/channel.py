"""The original shared, elementwise channel (not a full MIMO simulator)."""
import math

import torch
import torch.nn.functional as F


def response(batch, length, device, config):
    # Float distances avoid the int8 overflow in the original scripts.
    distance = torch.empty(batch, 1, device=device).uniform_(
        config['distance_min'], config['distance_max'])
    frequency = torch.arange(length, device=device) * config['subcarrier_spacing']
    phase = -2 * math.pi * distance / 3e8 * frequency
    attenuation = 1 + (torch.randn(batch, 2, device=device) /
                       torch.empty(batch, 2, device=device).uniform_(10, 50)).sum(1, keepdim=True)
    return attenuation * torch.complex(phase.cos(), phase.sin())


def complex_noise(reference, variance):
    # torch.randn_like(complex) has E[|z|^2] = 1, so each component has variance 1/2.
    return torch.randn_like(reference) * torch.as_tensor(variance, device=reference.device).sqrt()


def equalize(received, estimate):
    return received * estimate.conj() / estimate.abs().square().clamp_min(1e-8)


def normalize_power(x):
    return x / x.square().mean(dim=tuple(range(1, x.ndim)), keepdim=True).clamp_min(1e-8).sqrt()


def transmit(symbols, snr, target, ddpm, config):
    """symbols: [B, users, K]; returns recovered real symbols and CSI statistics."""
    batch, users, length = symbols.shape
    side = config['diffusion_side']
    padded_length = side * side
    if length > padded_length:
        raise ValueError(f'{length} symbols exceed the {side}x{side} diffusion grid')
    x = F.pad(symbols, (0, padded_length - length))
    h = response(batch, padded_length, symbols.device, config)
    signal = h[:, None] * x
    variance = signal[..., :length].abs().square().mean().detach() / (10 ** (snr / 10))
    h_initial = h + complex_noise(h, variance)
    received = signal + complex_noise(signal, variance)
    h_used = h_initial

    if target == 'h':
        parts = torch.cat((h_initial.real, h_initial.imag), dim=0).reshape(2 * batch, 1, side, side)
        parts = ddpm.denoise(parts, variance / 2).reshape(2 * batch, padded_length)
        h_used = torch.complex(parts[:batch], parts[batch:])
    elif target == 'y':
        flat = received.reshape(batch * users, side, side)
        parts = torch.stack((flat.real, flat.imag), dim=1)
        parts = ddpm.denoise(parts, variance / 2)
        received = torch.complex(parts[:, 0], parts[:, 1]).reshape(batch, users, padded_length)

    recovered = equalize(received, h_used[:, None]).real
    if target == 'x':
        # A scalar approximation, retained from the original X-oriented experiment.
        # Post-equalization errors are not actually iid Gaussian; document this limitation.
        recovered = ddpm.denoise(recovered.reshape(batch * users, 1, side, side), variance)
        recovered = recovered.reshape(batch, users, padded_length)
    stats = {
        'initial_error': (h_initial - h).abs().square().sum().detach(),
        'enhanced_error': (h_used - h).abs().square().sum().detach(),
        'channel_power': h.abs().square().sum().detach(),
    }
    return recovered[..., :length], stats
