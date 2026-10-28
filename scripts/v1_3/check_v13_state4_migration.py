"""审计 V1.3 四维状态迁移是否只新增了 gripper_open 输入列。"""

from pathlib import Path

import torch


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="checkpoints/v1.3_three_color_balanced_ft_seed8132026.pt")
    parser.add_argument("--out", default="checkpoints/v1.3_state4_init.pt")
    args = parser.parse_args()
    old = torch.load(Path(args.src), map_location="cpu", weights_only=False)
    new = torch.load(Path(args.out), map_location="cpu", weights_only=False)
    assert old["state_dim"] == 3
    assert new["state_dim"] == 4
    old_w = old["model_state"]["state_encoder.mlp.0.weight"]
    new_w = new["model_state"]["state_encoder.mlp.0.weight"]
    assert torch.equal(new_w[:, :3], old_w)
    assert torch.count_nonzero(new_w[:, 3]) == 0
    for key, value in old["model_state"].items():
        if key != "state_encoder.mlp.0.weight":
            assert torch.equal(value, new["model_state"][key]), key
    print("迁移审计通过：前三维状态及其余参数原样继承")
    print("新增 gripper_open 输入列为零初始化；state_dim=4")


if __name__ == "__main__":
    main()
