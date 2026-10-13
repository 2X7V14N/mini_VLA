import torch
import torch.nn as nn

class SimpleCharTokenizer:
    def __init__(self, vocabulary: list[str], max_length: int = 32) -> None:
        # ① 去重，但保留原顺序
        #    提示：dict 的 key 唯一且保持插入顺序
        unique = list(dict.fromkeys(vocabulary))

        # ② 两个特殊 token 的 ID，固定不变
        self.pad_id = 0
        self.unk_id = 1
        self.max_length = max_length

        # ③ 建"字符 -> ID"查表。PAD 和 UNK 已占 0、1，字符从 2 开始
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for index , char in enumerate(unique):
            self.stoi[char] = index + 2

        # ④ 词表大小
        #    提示：ID 从 0 开始，所以大小 = 最大 ID + 1
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> torch.Tensor:
        # ① 类型检查：不是字符串就抛 ValueError
        if not isinstance(text, str):
            raise ValueError("输入必须是字符串类型！")
        # ② 逐个字符转 ID
        #    提示：dict.get(key, 默认值) —— 认识就返回 ID，不认识返回 unk_id
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]

        # ③ 截断：只保留前 max_length 个
        ids = ids[:self.max_length]

        # ④ 补齐：用 pad_id 补到 max_length
        #    提示：[self.pad_id] * 还需要几个，然后两个列表相加
        ids = ids + [self.pad_id] * (self.max_length - len(ids))

        # ⑤ 转张量。dtype 必须是 torch.long
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, texts: list[str]) -> torch.Tensor:
    # 提示：每个元素调用 self.encode，得到一串 (max_length,) 张量
    #       再用 torch.stack 堆成 (B, max_length)
        return torch.stack([self.encode(t) for t in texts])

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, max_length: int = 32,
                 embed_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_id = 0

        # ① token embedding
        #    三个参数：num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0
        self.token_embed = nn.Embedding(
                                        num_embeddings=vocab_size, 
                                        embedding_dim=embed_dim, 
                                        padding_idx=0
                                        )

        # ② 可学习位置编码，形状 (1, max_length, embed_dim)
        #    ⚠️ 必须用 nn.Parameter，普通 tensor 不会更新
        self.pos_embed = nn.Parameter(torch.zeros(1, 
                                                  max_length, 
                                                  embed_dim
                                                  )
                                        )

        # ③ 一层 Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,          # 128
            nhead=num_heads,            # 4
            dim_feedforward=256,  # 256
            activation="gelu",       # "gelu"
            batch_first=True,      # ⚠️ 关键
        )

        # ④ 叠 num_layers 层
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # ① 校验：必须是二维 (B, L)
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids 应为二维, 实际 {token_ids.dim()} 维")

        # ② 校验：序列长度不能超过 max_length
        if token_ids.size(1) > self.max_length:
            raise ValueError(f"序列长度不能超过 max_length")

        # ③ embedding：(B, L) -> (B, L, embed_dim)
        x = self.token_embed(token_ids)

        # ④ 加位置编码（broadcasting）
        x = x + self.pos_embed[:, : x.size(1), :]

        # ⑤ 构造 padding mask
        #    ⚠️ PyTorch 的 src_key_padding_mask 里，True 表示"屏蔽掉"
        #    所以 PAD 的位置应该是 True
        padding_mask = (token_ids == self.pad_id)  # 形状 (B, L)，类型为 torch.bool

        # ⑥ 传给 Transformer
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        return x




       