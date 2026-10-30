"""Shared networks extracted from ResNet50withChannel and MFwithChannel."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .channel import response, complex_noise, equalize, normalize_power, transmit
from .diffusion import Diffusion
from .mfnet_backbone import MFNet


class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.bn = nn.BatchNorm2d(outputs)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class ChannelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnLeakyRelu2d(2048, 1024)
        self.conv2 = ConvBnLeakyRelu2d(1024, 256)
        self.conv3 = ConvBnLeakyRelu2d(256, 16)

    def forward(self, x):
        return normalize_power(self.conv3(self.conv2(self.conv1(x))).flatten(1).unsqueeze(1))


class ChannelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBnLeakyRelu2d(32, 256)
        self.conv2 = ConvBnLeakyRelu2d(256, 1024)
        self.conv3 = ConvBnLeakyRelu2d(1024, 2048)

    def forward(self, x, feature_shape):
        x = x.reshape(x.shape[0], 32, *feature_shape)
        return self.conv3(self.conv2(self.conv1(x)))


class SemanticSystem(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.target = config['target']
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        rgb, ir = resnet50(weights=weights), resnet50(weights=weights)
        ir.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.resnet_rgb = nn.Sequential(*list(rgb.children())[:-2])
        self.resnet_ir = nn.Sequential(*list(ir.children())[:-2])
        self.channel_encoder = ChannelEncoder()
        self.channel_decoder = ChannelDecoder()
        self.ddpm = Diffusion(self.target, config['diffusion']) if self.target != 'none' else None
        channels = (2048, 1024, 512, 256, 128, 64)
        for index, (inputs, outputs) in enumerate(zip(channels, channels[1:]), 1):
            setattr(self, f'up{index}', nn.Sequential(
                nn.ConvTranspose2d(inputs, outputs, 4, stride=2, padding=1),
                nn.BatchNorm2d(outputs), nn.ReLU(inplace=True)))
        self.final = nn.Conv2d(64, config['classes'], 1)
        self.channel_stats = {}

    def forward(self, images, snr=20, clean=False):
        rgb, ir = self.resnet_rgb(images[:, :3]), self.resnet_ir(images[:, 3:])
        x = torch.cat((self.channel_encoder(rgb), self.channel_encoder(ir)), dim=1)
        self.channel_stats = {}
        if not clean:
            channel_config = dict(self.config['channel'], diffusion_side=self.config['diffusion']['side'])
            x, self.channel_stats = transmit(x, snr, self.target, self.ddpm, channel_config)
        x = self.channel_decoder(x, rgb.shape[-2:])
        for index in range(1, 6):
            x = getattr(self, f'up{index}')(x)
        return self.final(x)

    def set_stage(self, stage):
        self.requires_grad_(False)
        names = ('resnet_rgb', 'resnet_ir', 'channel_encoder') if stage == 'stage1' else ()
        names += ('channel_decoder', 'up1', 'up2', 'up3', 'up4', 'up5', 'final')
        for name in names:
            getattr(self, name).requires_grad_(True)
        self.train_for_stage(stage)

    def train_for_stage(self, stage):
        self.train()
        if self.ddpm is not None:
            self.ddpm.eval()
        if stage == 'stage3':
            # Freezing parameters alone does not freeze BatchNorm running statistics.
            for name in ('resnet_rgb', 'resnet_ir', 'channel_encoder'):
                getattr(self, name).eval()


class MFNetChannel(MFNet):
    def __init__(self, config):
        super().__init__(config['classes'])
        self.config = config
        self.ddpm = None
        self.channel_stats = {}

    def forward(self, images, snr=20, clean=False):
        features = []
        for name, x in (('rgb', images[:, :3]), ('inf', images[:, 3:])):
            x = F.max_pool2d(getattr(self, f'conv1_{name}')(x), 2)
            p2 = getattr(self, f'conv2_2_{name}')(getattr(self, f'conv2_1_{name}')(x))
            p3 = getattr(self, f'conv3_2_{name}')(getattr(self, f'conv3_1_{name}')(F.max_pool2d(p2, 2)))
            p4 = getattr(self, f'conv4_{name}')(F.max_pool2d(p3, 2))
            p5 = getattr(self, f'conv5_{name}')(F.max_pool2d(p4, 2))
            features.append((p2, p3, p4, p5))
        fused = [torch.cat(parts, dim=1) for parts in zip(*features)]
        self.channel_stats = {}
        if not clean:
            # Reuse the same channel across scales, in blocks as in MFwithChannel.
            length = fused[-2].shape[-2] * fused[-2].shape[-1]
            h = response(images.shape[0], length, images.device, self.config['channel'])
            reference = fused[-1].flatten(2)
            repeated = h[:, None].repeat(1, 1, (reference.shape[-1] + length - 1) // length)[..., :reference.shape[-1]]
            variance = (reference * repeated).abs().square().mean().detach() / (10 ** (snr / 10))
            estimate = h + complex_noise(h, variance)
            recovered = []
            for feature in fused:
                flat = feature.flatten(2)
                repeats = (flat.shape[-1] + length - 1) // length
                hf = h[:, None].repeat(1, 1, repeats)[..., :flat.shape[-1]]
                ef = estimate[:, None].repeat(1, 1, repeats)[..., :flat.shape[-1]]
                signal = flat * hf
                y = signal + complex_noise(signal, variance)
                recovered.append(equalize(y, ef).real.reshape_as(feature))
            fused = recovered
        p2, p3, p4, x = fused
        for block, skip in ((self.decode4, p4), (self.decode3, p3), (self.decode2, p2)):
            x = block(F.interpolate(x, size=skip.shape[-2:], mode='nearest') + skip)
        return self.decode1(F.interpolate(x, size=images.shape[-2:], mode='nearest'))

    def set_stage(self, stage):
        self.requires_grad_(stage == 'stage1')
        if stage == 'stage3':
            for name in ('decode1', 'decode2', 'decode3', 'decode4'):
                getattr(self, name).requires_grad_(True)
        self.train_for_stage(stage)

    def train_for_stage(self, stage):
        self.train()
        if stage == 'stage3':
            for name, module in self.named_children():
                if name.startswith('conv'):
                    module.eval()


def build_model(config, pretrained=False):
    return MFNetChannel(config) if config['backbone'] == 'mfnet' else SemanticSystem(config, pretrained)
