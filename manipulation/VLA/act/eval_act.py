"""ACT 策略推理回放，闭环控制 MuJoCo 仿真。"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "simulators" / "mujoco"))
from utils.simulator import MujocoSimulator

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features


class ACTEvaluator:
    """ACT 推理器，在 MuJoCo 中闭环执行策略。"""

    def __init__(self, config_path="config.yaml"):
        """加载配置、策略模型和仿真器。"""
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        eval_cfg = self.cfg["eval"]
        ds_cfg = self.cfg["dataset"]
        cam_cfg = self.cfg["camera"]

        root = str(Path(config_path).parent / ds_cfg["root"])
        ckpt_dir = str(Path(config_path).parent / eval_cfg["ckpt_dir"])

        # 读取数据集元信息，构建 features
        meta = LeRobotDatasetMetadata(ds_cfg["repo_id"], root=root)
        policy_features = dataset_to_policy_features(meta.features)
        input_features = {k: v for k, v in policy_features.items() if k.startswith("observation.")}
        output_features = {k: v for k, v in policy_features.items() if v.type == FeatureType.ACTION}

        # ACT 配置（推理用，n_action_steps=1）
        act_cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=self.cfg["train"]["chunk_size"],
            n_action_steps=1,
            temporal_ensemble_coeff=eval_cfg["temporal_ensemble_coeff"],
        )

        # 加载策略
        self.policy = ACTPolicy.from_pretrained(ckpt_dir, config=act_cfg, dataset_stats=meta.stats)
        self.policy.to(eval_cfg["device"])
        self.policy.eval()

        # 仿真器
        sim_cfg_path = str(Path(config_path).parent / self.cfg["simulator_config"])
        self.sim = MujocoSimulator(sim_cfg_path)
        self.model = self.sim.model
        self.data = self.sim.data

        # 渲染器
        self.renderer = mujoco.Renderer(self.model, cam_cfg["height"], cam_cfg["width"])
        self.camera_name = cam_cfg["name"]
        self.img_key = f"observation.images.{cam_cfg['name']}"
        self.wrist_img_key = "observation.images.wrist"

        # 腕部相机
        self.wrist_cam = mujoco.MjvCamera()
        self.wrist_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.wrist_cam.trackbodyid = self.model.body("gripper").id
        self.wrist_cam.distance = 0.15
        self.wrist_cam.elevation = -90
        self.wrist_cam.azimuth = 0

        self.device = eval_cfg["device"]
        self.max_steps = eval_cfg["max_steps"]
        self.num_joints = self.model.nu

    def _render(self, camera):
        """渲染指定相机的图像。"""
        self.renderer.update_scene(self.data, camera)
        return self.renderer.render()

    def _to_tensor(self, img):
        """图像转为 CHW float tensor。"""
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _get_obs(self):
        """获取当前观测：双相机图像 + 关节状态。"""
        agent_img = self._render(self.camera_name)
        wrist_img = self._render(self.wrist_cam)
        state = torch.from_numpy(self.data.qpos[:self.num_joints].copy().astype(np.float32)).float()
        return {
            self.img_key: self._to_tensor(agent_img).unsqueeze(0).to(self.device),
            self.wrist_img_key: self._to_tensor(wrist_img).unsqueeze(0).to(self.device),
            "observation.state": state.unsqueeze(0).to(self.device),
        }

    def run(self):
        """主循环：20Hz 闭环推理。"""
        self.sim.start()
        dt = 1.0 / self.cfg["dataset"]["fps"]
        print(f"推理启动，最多 {self.max_steps} 步")

        with torch.no_grad():
            for step in range(self.max_steps):
                if not self.sim.is_running():
                    break

                t0 = time.time()

                # 锁内读取状态
                self.sim.locker.acquire()
                obs = self._get_obs()
                self.sim.locker.release()

                # 策略推理
                action = self.policy.select_action(obs)
                action_np = action.cpu().numpy().flatten()[:self.num_joints]

                # 锁内执行动作
                self.sim.locker.acquire()
                self.data.ctrl[:] = action_np
                self.sim.locker.release()

                # 控制频率
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

                if (step + 1) % 50 == 0:
                    print(f"Step {step + 1}/{self.max_steps}")

        self.sim.wait()
        print("推理结束")


if __name__ == "__main__":
    ACTEvaluator().run()
