"""ACT 策略训练脚本，基于 lerobot ACTPolicy。"""
import sys
from pathlib import Path

import torch
import yaml

from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.utils import dataset_to_policy_features


class ACTTrainer:
    """ACT 训练器，从 LeRobotDataset 加载数据训练 ACTPolicy。"""

    def __init__(self, config_path="config.yaml"):
        """加载配置，构建 ACTConfig 和数据集。"""
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        train_cfg = self.cfg["train"]
        ds_cfg = self.cfg["dataset"]
        root = str(Path(config_path).parent / ds_cfg["root"])

        # 读取数据集元信息
        self.meta = LeRobotDatasetMetadata(ds_cfg["repo_id"], root=root)

        # 构建 policy features
        policy_features = dataset_to_policy_features(self.meta.features)
        input_features = {k: v for k, v in policy_features.items() if k.startswith("observation.")}
        output_features = {k: v for k, v in policy_features.items() if v.type == FeatureType.ACTION}

        # ACT 配置
        self.act_cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=train_cfg["chunk_size"],
            n_action_steps=train_cfg["chunk_size"],
        )

        # delta timestamps
        delta_timestamps = resolve_delta_timestamps(self.act_cfg, self.meta)

        # 加载数据集
        self.dataset = LeRobotDataset(
            ds_cfg["repo_id"],
            root=root,
            delta_timestamps=delta_timestamps,
        )

        # 训练参数
        self.device = train_cfg["device"]
        self.batch_size = train_cfg["batch_size"]
        self.num_workers = train_cfg["num_workers"]
        self.lr = train_cfg["lr"]
        self.training_steps = train_cfg["training_steps"]
        self.log_freq = train_cfg["log_freq"]
        self.output_dir = str(Path(config_path).parent / train_cfg["output_dir"])

    def train(self):
        """执行训练循环。"""
        policy = ACTPolicy(self.act_cfg, dataset_stats=self.meta.stats)
        policy.train()
        policy.to(self.device)

        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

        print(f"训练 {self.training_steps} 步, batch_size={self.batch_size}, device={self.device}")
        data_iter = iter(dataloader)
        for step in range(1, self.training_steps + 1):
            # 循环取数据
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # 移到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            loss, loss_dict = policy.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % self.log_freq == 0 or step == 1:
                loss_val = loss.item()
                print(f"Step {step}/{self.training_steps}  loss={loss_val:.4f}")

        # 保存模型
        policy.save_pretrained(self.output_dir)
        print(f"模型已保存到 {self.output_dir}")


if __name__ == "__main__":
    ACTTrainer().train()
