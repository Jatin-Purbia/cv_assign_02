"""Small original encoder-decoder FCNs; no model zoo or external weights."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, separable: bool = False,
                 kernel: int = 3):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError('kernel must be positive and odd')
        if separable:
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, incoming, kernel, padding=kernel//2,
                          groups=incoming, bias=False),
                nn.BatchNorm2d(incoming), nn.ReLU(inplace=True),
                nn.Conv2d(incoming, outgoing, 1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True))
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, outgoing, kernel, padding=kernel//2, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SmallFCN(nn.Module):
    """Configurable depth, widths, concatenated skips and bilinear/transpose decoder."""
    def __init__(self, kind: str = 'tiny', base: int = 8, depth: int = 3,
                 decoder_width: int = 0, kernel: int = 3,
                 upsampling: str = 'bilinear', skips: bool = True):
        super().__init__()
        if kind not in ('tiny', 'unet', 'depthwise') or not 2 <= depth <= 4 or base < 2:
            raise ValueError('Invalid architecture kind, depth or base width')
        if upsampling not in ('bilinear', 'transpose') or decoder_width < 0:
            raise ValueError('Invalid decoder configuration')
        self.skips, self.upsampling = skips, upsampling
        widths = [base * 2**i for i in range(depth)]
        sep = kind == 'depthwise'
        self.encoder = nn.ModuleList()
        incoming = 3
        for width in widths:
            blocks = [ConvBlock(incoming, width, sep, kernel)]
            if kind == 'unet':
                blocks.append(ConvBlock(width, width, False, kernel))
            self.encoder.append(nn.Sequential(*blocks))
            incoming = width
        self.decoder, self.up = nn.ModuleList(), nn.ModuleList()
        for width in reversed(widths[:-1]):
            output = decoder_width or width
            self.up.append(nn.ConvTranspose2d(incoming, incoming, 2, 2)
                           if upsampling == 'transpose' else nn.Identity())
            self.decoder.append(ConvBlock(incoming + (width if skips else 0), output, sep, kernel))
            incoming = output
        self.head = nn.Conv2d(incoming, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i, block in enumerate(self.encoder):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            features.append(x)
        for block, up, skip in zip(self.decoder, self.up, reversed(features[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            if self.skips:
                x = torch.cat((x, skip), dim=1)
            x = block(x)
        return self.head(x)


def build_model(config: dict) -> SmallFCN:
    return SmallFCN(**config)
