"""V1.5a model: V1.3 policy plus causal visual history and zone heads."""

import torch
from torch import nn

from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color


class LanguageConditionedMiniVLA3ColorDynamic(LanguageConditionedMiniVLA3Color):
    """Keep the V1.3 action path and add trainable temporal/zone predictions."""

    def __init__(self, *args, history=4, **kwargs):
        super().__init__(*args, **kwargs)
        if history < 1:
            raise ValueError("history 必须为正整数")
        self.history = int(history)
        embed_dim = self.text_encoder.token_embed.embedding_dim
        self.temporal_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.temporal_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)
        )
        # Preserve the validated V1.3 policy at initialization.  The temporal
        # pathway is learned as a residual instead of immediately perturbing
        # every visual token with random GRU features.
        self.temporal_gate = nn.Parameter(torch.zeros(()))
        self.zone_position_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 2), nn.Tanh()
        )
        self.zone_velocity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 2), nn.Tanh()
        )

    def _conditioned(self, image_history, token_ids, robot_state):
        if image_history.dim() == 4:
            image_history = image_history.unsqueeze(1)
        if image_history.dim() != 5:
            raise ValueError("image_history 必须为 (B,T,3,H,W)")
        b, t = image_history.shape[:2]
        if t < 1:
            raise ValueError("图像历史不能为空")
        flat = image_history.reshape(b * t, *image_history.shape[2:])
        vision_tokens = self.vision_encoder(flat).reshape(b, t, -1, self.fusion.input_norm.normalized_shape[0])
        pooled = vision_tokens.mean(dim=2)
        temporal, _ = self.temporal_gru(pooled)
        temporal_summary = self.temporal_proj(temporal[:, -1])
        current_vision = vision_tokens[:, -1] + (
            torch.tanh(self.temporal_gate) * temporal_summary.unsqueeze(1)
        )

        text_tokens = self.text_encoder(token_ids)
        state_token = self.state_encoder(robot_state)
        padding = token_ids == 0
        fused = self.fusion(current_vision, text_tokens, state_token, padding)
        text_tokens = text_tokens.masked_fill(padding.unsqueeze(-1), 0.0)
        lang = self.language_skip(text_tokens.flatten(start_dim=1))
        block_color, zone_color = self._color_roles(token_ids)
        conditioned = self.condition_fusion(torch.cat([
            fused, lang, self.block_color_embed(block_color),
            self.zone_color_embed(zone_color),
        ], dim=-1))
        return conditioned

    def forward(self, image_history, token_ids, robot_state, return_aux=False):
        conditioned = self._conditioned(image_history, token_ids, robot_state)
        action = self.action_head(conditioned)
        if not return_aux:
            return action
        return {
            "action": action,
            "zone_position": self.zone_position_head(conditioned),
            # Velocity labels are normalized by 0.02 world units/tick.
            "zone_velocity": self.zone_velocity_head(conditioned),
        }

    @classmethod
    def from_v13_checkpoint(cls, checkpoint, tokenizer, history=4):
        model = cls(
            vocab_size=tokenizer.vocab_size,
            max_length=int(checkpoint["max_length"]),
            state_dim=int(checkpoint["state_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            chunk_size=int(checkpoint["chunk_size"]),
            history=history,
            red_token_id=tokenizer.stoi["红"],
            blue_token_id=tokenizer.stoi["蓝"],
            yellow_token_id=tokenizer.stoi["黄"],
        )
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state"], strict=False
        )
        expected_prefixes = ("temporal_gru.", "temporal_proj.",
                             "zone_position_head.", "zone_velocity_head.")
        incompatible_missing = [
            key for key in missing
            if key != "temporal_gate" and not key.startswith(expected_prefixes)
        ]
        if unexpected or incompatible_missing:
            raise ValueError(
                "V1.3 权重迁移异常 "
                f"missing={incompatible_missing} unexpected={list(unexpected)}"
            )
        return model
