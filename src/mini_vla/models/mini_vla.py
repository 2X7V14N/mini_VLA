import torch
import torch.nn as nn
from mini_vla.models.vision import VisionEncoder
from mini_vla.models.text import TextEncoder
from mini_vla.models.state import StateEncoder
from mini_vla.models.fusion import MultiModalFusion
from mini_vla.models.action import ActionHead


class MiniVLA(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 max_length: int = 32, state_dim: int = 2,
                 action_dim: int = 2, chunk_size: int = 4) -> None:
        super().__init__()

        self.vision_encoder = VisionEncoder()                      # ①
        self.text_encoder = TextEncoder(                           # ②
            vocab_size=vocab_size, max_length=max_length, embed_dim=embed_dim,
        )
        self.state_encoder = StateEncoder(                         # ③
            state_dim=state_dim, embed_dim=embed_dim,
        )
        self.fusion = MultiModalFusion(                            # ④
            embed_dim=embed_dim, num_vision_tokens=16,
        )
        self.action_head = ActionHead(                             # ⑤
            embed_dim=embed_dim, action_dim=action_dim, chunk_size=chunk_size,
        )

    def forward(self, image, token_ids, robot_state):
        vision_tokens = self.vision_encoder(image)      # (B, 16, 128)
        text_tokens   = self.text_encoder(token_ids)    # (B, L,  128)
        state_token   = self.state_encoder(robot_state) # (B, 1,  128)

        text_padding_mask = token_ids == 0              # (B, L)  ← 填什么？

        fused = self.fusion(vision_tokens, text_tokens, state_token, text_padding_mask)
        return self.action_head(fused)
