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

class LanguageConditionedMiniVLA(MiniVLA):
    """V1.2 policy with explicit object/zone color roles.

    The standard MiniVLA asks one learned fusion query to retain all language
    information alongside vision and state.  V1.2 instructions encode two
    independent roles (object color and destination-zone color), so this model
    also gives the action head a direct, position-aware text summary and role
    embeddings for the first and second color mentions (object, then zone).
    This makes the two instruction variables easy to learn without changing
    the image/state pathway. The original MiniVLA and its checkpoints remain
    unchanged.
    """

    def __init__(self, *args, red_token_id=None, blue_token_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        if red_token_id is None or blue_token_id is None:
            raise ValueError("V1.2 需要传入红色和蓝色 token ID")
        self.red_token_id = int(red_token_id)
        self.blue_token_id = int(blue_token_id)
        embed_dim = self.text_encoder.token_embed.embedding_dim
        self.language_skip = nn.Sequential(
            nn.Linear(self.text_encoder.max_length * embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.block_color_embed = nn.Embedding(2, embed_dim)
        self.zone_color_embed = nn.Embedding(2, embed_dim)
        self.condition_fusion = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def _color_roles(self, token_ids):
        """Return categorical IDs for the first (block) and second (zone) color.

        V1.2's supported instruction templates mention exactly two colors in
        order: the manipulated block first, destination zone second. Parse by
        token identity rather than absolute character offsets, so templates can
        vary in wording while preserving that contract.
        """
        is_color = ((token_ids == self.red_token_id)
                    | (token_ids == self.blue_token_id))
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        positions = positions.unsqueeze(0).expand_as(token_ids)
        first_pos = positions.masked_fill(~is_color, token_ids.size(1)).min(dim=1).values
        after_first = is_color & (positions > first_pos.unsqueeze(1))
        second_pos = positions.masked_fill(~after_first, token_ids.size(1)).min(dim=1).values
        if torch.any(first_pos == token_ids.size(1)) or torch.any(second_pos == token_ids.size(1)):
            raise ValueError("V1.2 指令必须包含方块色和区域色两个颜色 token")
        first_token = token_ids.gather(1, first_pos.unsqueeze(1)).squeeze(1)
        second_token = token_ids.gather(1, second_pos.unsqueeze(1)).squeeze(1)
        # Embedding index 0 = red, 1 = blue.
        block_color = (first_token == self.blue_token_id).long()
        zone_color = (second_token == self.blue_token_id).long()
        return block_color, zone_color

    def forward(self, image, token_ids, robot_state):
        vision_tokens = self.vision_encoder(image)
        text_tokens = self.text_encoder(token_ids)
        state_token = self.state_encoder(robot_state)
        text_padding_mask = token_ids == 0

        fused = self.fusion(vision_tokens, text_tokens, state_token,
                            text_padding_mask)
        # Preserve character position/role explicitly.  Padding is zeroed so
        # sentence length does not become a spurious conditioning signal.
        text_tokens = text_tokens.masked_fill(text_padding_mask.unsqueeze(-1), 0.0)
        text_flat = text_tokens.flatten(start_dim=1)
        lang = self.language_skip(text_flat)
        block_color, zone_color = self._color_roles(token_ids)
        block_role = self.block_color_embed(block_color)
        zone_role = self.zone_color_embed(zone_color)
        conditioned = self.condition_fusion(
            torch.cat([fused, lang, block_role, zone_role], dim=-1))
        return self.action_head(conditioned)
