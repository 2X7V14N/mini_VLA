import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, num_vision_tokens: int = 16, 
                 dim_feedforward: int = 256) -> None:
        super().__init__()
        self.num_vision_tokens = num_vision_tokens      # ← 存起来，mask 要用

        # ① 输入归一化（上面刚说的）
        self.input_norm = nn.LayerNorm(embed_dim)

        # ② 融合查询 token，形状 (1, 1, 128)
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ③ 四个模态类型嵌入，各自形状 (1, 1, 128)
        self.mod_query  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_vision = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_text   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_state  = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ④ Transformer（和 text.py 里一样的写法）
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,dim_feedforward = dim_feedforward,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, vision_tokens, text_tokens, state_token, text_padding_mask):
        B = vision_tokens.shape[0]

        # ① 归一化 + ② 加模态嵌入（一步完成）
        vision = self.input_norm(vision_tokens) + self.mod_vision
        text   = self.input_norm(text_tokens)   + self.mod_text
        state  = self.input_norm(state_token)   + self.mod_state
                # 查询 token 要和模态嵌入相加，然后扩展到批次大小
        query = (self.query_token + self.mod_query).expand(B,-1,-1)                            # 填什么？

        seq = torch.cat([query, vision, text, state], dim=1)

        L = text_padding_mask.shape[1]

        # 前三段：查询(1) + 视觉(16) + ... 都是 False
        # 提示：用 text_padding_mask.new_zeros(B, n) 造全 False，
        #       .new_zeros 会自动沿用 dtype 和 device，不用手写
        head = text_padding_mask.new_zeros(B, 1 + self.num_vision_tokens)   # (B, 17)
        tail = text_padding_mask.new_zeros(B, 1)                            # (B, 1)

        full_mask = torch.cat([head, text_padding_mask, tail], dim=1)                    # 中间填什么？

        out = self.encoder(seq, src_key_padding_mask=full_mask)    # (B, 18+L, 128)

        fused = out[:, 0]                                            # (B, 128)
        return fused



