import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int = 2, embed_dim: int = 128,
                 hidden_dim: int = 64) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),     # 空①
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),     # 空②
            nn.LayerNorm(embed_dim),        # 空③
        )

    def forward(self, robot_state: torch.Tensor) -> torch.Tensor:
        x = self.mlp(robot_state)
        return x.unsqueeze(1)                   # 空④
