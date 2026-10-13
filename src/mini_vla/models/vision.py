import torch
import torch.nn as nn

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

        x = self.stem(image)
        x = x.flatten(2)      # (B, 128, 4, 4) → (B, 128, 16)
        x = x.transpose(1, 2) # (B, 128, 16)  → (B, 16, 128)
        x = x + self.pos_embed

        return x
