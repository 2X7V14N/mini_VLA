"""V1.5c: zero-gated temporal V1.3 policy with both-object tracking heads."""

import torch
from torch import nn

from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color


class LanguageConditionedMiniVLA3ColorDynamicBoth(LanguageConditionedMiniVLA3Color):
    def __init__(self, *args, history=4, **kwargs):
        super().__init__(*args, **kwargs)
        if history < 1:
            raise ValueError("history 必须为正整数")
        self.history = int(history)
        embed = self.text_encoder.token_embed.embedding_dim
        self.temporal_gru = nn.GRU(embed, embed, batch_first=True)
        self.temporal_proj = nn.Sequential(nn.Linear(embed, embed), nn.GELU(), nn.LayerNorm(embed))
        self.temporal_gate = nn.Parameter(torch.zeros(()))
        self.block_position_head = self._head(embed)
        self.block_velocity_head = self._head(embed)
        self.zone_position_head = self._head(embed)
        self.zone_velocity_head = self._head(embed)
        self.future_zone_head = self._head(embed)

    @staticmethod
    def _head(embed):
        return nn.Sequential(nn.Linear(embed, embed), nn.GELU(), nn.Linear(embed, 2), nn.Tanh())

    def _conditioned(self, image_history, token_ids, robot_state):
        if image_history.dim() == 4:
            image_history = image_history.unsqueeze(1)
        if image_history.dim() != 5:
            raise ValueError("image_history 必须为 (B,T,3,H,W)")
        batch, ticks = image_history.shape[:2]
        flat = image_history.reshape(batch * ticks, *image_history.shape[2:])
        tokens = self.vision_encoder(flat).reshape(
            batch, ticks, -1, self.fusion.input_norm.normalized_shape[0]
        )
        temporal, _ = self.temporal_gru(tokens.mean(dim=2))
        residual = self.temporal_proj(temporal[:, -1])
        vision = tokens[:, -1] + torch.tanh(self.temporal_gate) * residual.unsqueeze(1)
        text = self.text_encoder(token_ids)
        state = self.state_encoder(robot_state)
        padding = token_ids == 0
        fused = self.fusion(vision, text, state, padding)
        text = text.masked_fill(padding.unsqueeze(-1), 0.0)
        language = self.language_skip(text.flatten(start_dim=1))
        block, zone = self._color_roles(token_ids)
        return self.condition_fusion(torch.cat([
            fused, language, self.block_color_embed(block), self.zone_color_embed(zone),
        ], dim=-1))

    def forward(self, image_history, token_ids, robot_state, return_aux=False):
        conditioned = self._conditioned(image_history, token_ids, robot_state)
        action = self.action_head(conditioned)
        if not return_aux:
            return action
        return {
            "action": action,
            "block_position": self.block_position_head(conditioned),
            "block_velocity": self.block_velocity_head(conditioned),
            "zone_position": self.zone_position_head(conditioned),
            "zone_velocity": self.zone_velocity_head(conditioned),
            "future_zone_position": self.future_zone_head(conditioned),
        }

    @classmethod
    def from_v13_checkpoint(cls, checkpoint, tokenizer, history=4):
        model = cls(
            vocab_size=tokenizer.vocab_size, max_length=int(checkpoint["max_length"]),
            state_dim=int(checkpoint["state_dim"]), action_dim=int(checkpoint["action_dim"]),
            chunk_size=int(checkpoint["chunk_size"]), history=history,
            red_token_id=tokenizer.stoi["红"], blue_token_id=tokenizer.stoi["蓝"],
            yellow_token_id=tokenizer.stoi["黄"],
        )
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        new = ("temporal_gru.", "temporal_proj.", "block_position_head.",
               "block_velocity_head.", "zone_position_head.",
               "zone_velocity_head.", "future_zone_head.")
        invalid = [key for key in missing if key != "temporal_gate" and not key.startswith(new)]
        if unexpected or invalid:
            raise ValueError(f"V1.3 权重迁移异常 missing={invalid} unexpected={list(unexpected)}")
        return model
