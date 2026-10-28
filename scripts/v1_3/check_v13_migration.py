from pathlib import Path

import torch


SRC = Path("checkpoints/v1.2_final.pt")
MIGRATED = Path("checkpoints/v1.3_init.pt")

old = torch.load(SRC, map_location="cpu", weights_only=False)
new = torch.load(MIGRATED, map_location="cpu", weights_only=False)

assert new["model_type"] == "LanguageConditionedMiniVLA3Color"
assert {"红", "蓝", "黄"}.issubset(set(new["vocab"]))

old_state = old["model_state"]
new_state = new["model_state"]

token_key = "text_encoder.token_embed.weight"
role_keys = ("block_color_embed.weight", "zone_color_embed.weight")
special = {token_key, *role_keys}

# 所有不需要扩展的参数应与 V1.2 完全相同。
checked = 0
for key, old_tensor in old_state.items():
    if key in special:
        continue
    assert key in new_state, f"迁移后缺少参数：{key}"
    assert torch.equal(old_tensor, new_state[key]), f"参数未原样继承：{key}"
    checked += 1

# 按字符对应的 token embedding 行应原样继承。
old_ids = {"<pad>": 0, "<unk>": 1}
old_ids.update({char: i + 2 for i, char in enumerate(old["vocab"])})
new_ids = {"<pad>": 0, "<unk>": 1}
new_ids.update({char: i + 2 for i, char in enumerate(new["vocab"])})

for token, old_id in old_ids.items():
    assert torch.equal(
        old_state[token_key][old_id],
        new_state[token_key][new_ids[token]],
    ), f"token embedding 未正确继承：{token!r}"

# 黄色 token embedding 应等于红、蓝 embedding 的均值初始化。
yellow_id = new_ids["黄"]
expected_yellow_token = (
    old_state[token_key][old_ids["红"]]
    + old_state[token_key][old_ids["蓝"]]
) / 2
assert torch.allclose(
    new_state[token_key][yellow_id],
    expected_yellow_token,
), "黄色 token embedding 初始化不符合预期"

# 角色嵌入行：红=0、蓝=1 应继承，黄=2 应为红蓝均值。
for key in role_keys:
    old_weight = old_state[key]
    new_weight = new_state[key]
    assert new_weight.shape[0] == 3, f"{key} 没扩展到三种颜色"
    assert torch.equal(new_weight[0], old_weight[0]), f"{key} 红色行不匹配"
    assert torch.equal(new_weight[1], old_weight[1]), f"{key} 蓝色行不匹配"
    assert torch.allclose(
        new_weight[2], old_weight.mean(dim=0)
    ), f"{key} 黄色行初始化不符合预期"

print(f"迁移审计通过：{checked} 个普通参数逐项继承")
print("红、蓝 token 与角色嵌入继承正确；黄色初始化正确")
print(f"模型类型：{new['model_type']}；词表字符数：{len(new['vocab'])}")
