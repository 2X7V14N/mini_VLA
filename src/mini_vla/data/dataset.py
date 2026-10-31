import torch
from torch.utils.data import Dataset


class RobotDataset(Dataset):
    """一条样本 = 一帧（图像 + 指令 + 状态 + 动作块）。"""

    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        item = {
            "image":torch.from_numpy(s["image"]).permute(2, 0, 1).float() / 255.0,     # (3, 64, 64) float32
            "token_ids":self.tokenizer.encode(s["instruction"]),     # (L,) long
            "robot_state":torch.from_numpy(s["robot_state"]),   # (2,) float32
            "action_chunk": torch.from_numpy(s["action_chunk"])    # (4, 2) float32
        }
        if "phase_id" in s:
            item["phase_id"] = torch.tensor(s["phase_id"], dtype=torch.long)
        return item
