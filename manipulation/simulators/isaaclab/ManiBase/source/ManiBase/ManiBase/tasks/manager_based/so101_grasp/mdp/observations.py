"""SO101 抓取任务观测函数。"""

from __future__ import annotations

import torch
from isaaclab.envs.mdp import ManagerTermBase
from isaaclab.managers import SceneEntityCfg


def object_position_in_robot_root_frame(env, robot_cfg, object_cfg) -> torch.Tensor:
    """计算物体相对机器人根位置的向量。"""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, :3] - robot.data.body_pos_w[:, 0, :3]


def ee_to_object_vector(env, ee_frame_cfg, object_cfg, ee_frame_index: int = 0) -> torch.Tensor:
    """末端执行器指向物体的向量（世界系）。"""
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :]
    return obj.data.root_pos_w[:, :3] - ee_pos


def ee_position_world(env, ee_frame_cfg, ee_frame_index: int = 0) -> torch.Tensor:
    """末端执行器的世界坐标。"""
    ee_frame = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, ee_frame_index, :]


class ResNet10Extractor(ManagerTermBase):
    """相机图像 ResNet10 特征提取器。

    CPU 推理避免与 Isaac Sim GPU 内存冲突。输出 (B, 512) 全局平均池化特征。
    """

    def __init__(self, cfg, env):
        from transformers import AutoModel

        super().__init__(cfg, env)
        model_name = getattr(cfg, "model_name", "helper2424/resnet10")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cpu").eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cpu")
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cpu")
        self._device = env.device

    def __call__(self, env, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
        from isaaclab.envs.mdp.observations import image

        image_data = image(env=env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False)
        image_data = image_data.permute(0, 3, 1, 2).float().cpu() / 255.0
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        image_data = (image_data - mean) / std
        with torch.no_grad():
            out = self.model(image_data)
            features = out.last_hidden_state.mean(dim=[2, 3])  # (B, 512)
        return features.to(self._device)
