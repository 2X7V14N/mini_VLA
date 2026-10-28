import torch
from torch import nn

from mini_vla.models.mini_vla import LanguageConditionedMiniVLA


class LanguageConditionedMiniVLA3Color(LanguageConditionedMiniVLA):
    """V1.3 三色模型：红、蓝、黄方块与放置区。"""

    def __init__(self, *args, yellow_token_id=None, **kwargs):
        if yellow_token_id is None:
            raise ValueError("V1.3 需要传入黄色 token ID")

        # 先创建与 V1.2 相同的网络结构。
        super().__init__(*args, **kwargs)

        self.yellow_token_id = int(yellow_token_id)

        # V1.2 的颜色嵌入有两行；V1.3 扩展为红、蓝、黄三行。
        # 旧权重稍后会通过迁移代码复制，当前这两层暂时随机初始化。
        embed_dim = self.text_encoder.token_embed.embedding_dim
        self.block_color_embed = nn.Embedding(3, embed_dim)
        self.zone_color_embed = nn.Embedding(3, embed_dim)

    def _color_roles(self, token_ids):
        """从指令中依次提取方块颜色和放置区颜色。

        颜色编号固定为：红=0、蓝=1、黄=2。
        指令必须先提到方块颜色，再提到放置区颜色。
        """
        is_color = (
            (token_ids == self.red_token_id)
            | (token_ids == self.blue_token_id)
            | (token_ids == self.yellow_token_id)
        )

        positions = torch.arange(
            token_ids.size(1), device=token_ids.device
        ).unsqueeze(0).expand_as(token_ids)

        first_pos = positions.masked_fill(
            ~is_color, token_ids.size(1)
        ).min(dim=1).values

        after_first = is_color & (positions > first_pos.unsqueeze(1))
        second_pos = positions.masked_fill(
            ~after_first, token_ids.size(1)
        ).min(dim=1).values

        missing_color = (
            (first_pos == token_ids.size(1))
            | (second_pos == token_ids.size(1))
        )
        if torch.any(missing_color):
            raise ValueError("指令必须包含方块颜色和放置区颜色")

        first_token = token_ids.gather(
            1, first_pos.unsqueeze(1)
        ).squeeze(1)
        second_token = token_ids.gather(
            1, second_pos.unsqueeze(1)
        ).squeeze(1)

        def color_id(token):
            result = torch.full_like(token, -1)
            result[token == self.red_token_id] = 0
            result[token == self.blue_token_id] = 1
            result[token == self.yellow_token_id] = 2
            return result

        block_color = color_id(first_token)
        zone_color = color_id(second_token)

        if torch.any(block_color < 0) or torch.any(zone_color < 0):
            raise ValueError("发现未支持的方块色或放置区颜色")

        return block_color, zone_color