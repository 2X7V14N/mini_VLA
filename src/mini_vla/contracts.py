from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    """策略在一个时刻接收到的观测。"""

    image: np.ndarray
    instruction: str
    robot_state: np.ndarray

    def __post_init__(self) -> None:
        # TODO 1：检查 image 的形状是否为 (64, 64, 3)
        if not isinstance(self.image, np.ndarray):
            raise ValueError("image 应为 numpy.ndarray")

        if not self.image.shape == (64, 64, 3):
            raise ValueError(
                f"image 形状应为 (64,64,3)，实际为 {self.image.shape}"
            )     
        # TODO 2：检查 image 的 dtype 是否为 np.uint8
        if not self.image.dtype == np.uint8:
            raise ValueError(
                f"image 的 dtype 应该为 np.uint8"
            )   
        # TODO 3：检查 instruction 是否为非空字符串
        if not isinstance(self.robot_state, np.ndarray):
            raise ValueError("robot_state 应为 numpy.ndarray")

        if  not isinstance(self.instruction, str):
            raise ValueError(
                f"instruction 应为字符串"
            )   
        elif  len(self.instruction.strip()) == 0:
            raise ValueError(
                f"instruction 为空字符串"
            )
        # TODO 4：检查 robot_state 的形状是否为 (2,)
        if  self.robot_state.shape != (2,):
            raise ValueError(
                f"robot_state 的形状应为 (2,)"
            )
        # TODO 5：检查 robot_state 是否为有限的数值，并且范围在 [-1, 1]
        # 不符合约定时，用 raise ValueError(...) 报错
        if  not np.isfinite(self.robot_state).all():  
            raise ValueError(
                f"robot_state 应为有限的数值"
            )
        
        if  self.robot_state.min() < -1 or self.robot_state.max() > 1:
            raise ValueError(
                f"robot_state 的范围应该在 [-1, 1]"
            ) 





@dataclass
class Action:
    """策略在一个时刻输出的二维移动动作。"""

    delta_xy: np.ndarray

    def __post_init__(self) -> None:
        # TODO 1：检查 delta_xy 的形状是否为 (2,)
        if not isinstance(self.delta_xy, np.ndarray):
            raise ValueError("delta_xy 应为 numpy.ndarray") 
        if not self.delta_xy.shape == (2,):
            raise ValueError(
                f"delta_xy 的形状应为 (2,)"
            )
        # TODO 2：检查它是否为有限数值，并且范围在 [-1, 1]
        # 不符合约定时，用 raise ValueError(...) 报错
        if not np.isfinite(self.delta_xy).all():
            raise ValueError(
                f"delta_xy 应该为有限的数值"
            )  
        if  self.delta_xy.min() < -1 or self.delta_xy.max() > 1:
            raise ValueError(
                f"delta_xy 的范围应该在 [-1, 1]"
            ) 