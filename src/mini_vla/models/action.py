import torch
import torch.nn as nn

class ActionHead(nn.Module):
    def __init__(self, embed_dim: int = 128, action_dim: int = 2,
                 chunk_size: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.chunk_size = chunk_size          # 存起来，reshape 要用
        self.action_dim = action_dim          # 存起来，reshape 要用

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),                    # ①
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, chunk_size * action_dim),                         # ② 填什么？
            nn.Tanh(),
        )

    def forward(self, fused_feature: torch.Tensor) -> torch.Tensor:
        x = self.mlp(fused_feature)      # (B, 128) -> (B, 8)
        return x.view(-1, self.chunk_size, self.action_dim)                      # (B, 8) -> (B, 4, 2)


