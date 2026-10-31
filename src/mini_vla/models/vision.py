import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, 128))
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )


    def forward(self,image):
        tokens, _ = self.forward_with_intermediate(image)
        return tokens

    def forward_with_intermediate(self, image):
        """Return final tokens and the shared 8x8 feature map for auxiliary heads."""
        x = image
        high_resolution = None
        for index, layer in enumerate(self.stem):
            x = layer(x)
            if index == 5:
                high_resolution = x
        h, w = x.shape[-2:]
        pos = self.pos_embed.transpose(1, 2).reshape(1, 128, 4, 4)
        if (h, w) != (4, 4):
            pos = F.interpolate(pos, size=(h, w), mode="bicubic",
                                align_corners=False)
        pos = pos.flatten(2).transpose(1, 2)
        x = x.flatten(2)      # (B, 128, H, W) → (B, 128, H*W)
        x = x.transpose(1, 2) # (B, 128, 16)  → (B, 16, 128)
        x = x + pos

        return x, high_resolution
