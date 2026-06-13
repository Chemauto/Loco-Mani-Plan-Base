"""SO101 抓取任务集成冒烟测试（headless）。

验证：jaw link prim 有效、ContactSensor 接线、观测维度一致、奖励 finite，
并通过"把 cube 主动压到夹爪指"确认接触检测真正生效（随机动作通常碰不到 cube）。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ManiBase.tasks  # noqa: F401
from ManiBase.robots import SO101_JAW_LINK_NAME

TASK = "ManiBase-SO101-Grasp-Cube-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()

    policy_dim = env.observation_space["policy"].shape[-1]
    # joint_pos(6) + joint_vel(6) + gripper_pos(1) + ee_to_cube(3) + ee_pos(3) + actions(6) = 25
    print(f"[smoke] policy obs dim = {policy_dim}  (expect 25)")

    # 随机动作跑一段，确认能步进、obs/reward finite
    min_r, max_r = float("inf"), float("-inf")
    obs = None
    for _ in range(args.steps):
        with torch.inference_mode():
            a = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            obs, reward, *_ = env.step(a)
            min_r, max_r = min(min_r, reward.min().item()), max(max_r, reward.max().item())
    obs_finite = torch.isfinite(obs["policy"]).all().item()
    rew_finite = bool(torch.isfinite(torch.tensor([min_r, max_r])).all())
    print(f"[smoke] random phase: obs finite={obs_finite}, reward[{min_r:.4f},{max_r:.4f}] finite={rew_finite}")

    # 把 cube 贴到夹爪指并命令夹爪闭合，确认 ContactSensor 上报 finger<->cube 接触
    contact_max = 0.0
    with torch.inference_mode():
        cube = env.unwrapped.scene["cube"]
        robot = env.unwrapped.scene["robot"]
        jaw_idx = robot.data.body_names.index(SO101_JAW_LINK_NAME)
        close_act = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        close_act[:, -1] = -1.0  # 二值夹爪闭合
        for _ in range(15):
            root_state = cube.data.root_state_w.clone()
            root_state[:, :3] = robot.data.body_pos_w[:, jaw_idx, :]
            root_state[:, 7:13] = 0.0
            cube.write_root_state_to_sim(root_state)
            env.step(close_act)
            sensor = env.unwrapped.scene["jaw_contact_forces"]
            contact_max = max(contact_max, torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1).max().item())
    print(f"[smoke] contact force (cube at jaw + gripper closing): max={contact_max:.3f} N (config threshold=0.3)")

    # 断言（reward 项的 SceneEntityCfg 已在 env 构造/步进时由 RewardManager 正确解析，
    # 这里直接读传感器力，确认 finger<->cube 接触检测链路有效）
    assert policy_dim == 25, f"obs dim {policy_dim} != 25"
    assert obs_finite and rew_finite, "obs/reward non-finite"
    assert contact_max > 0.1, f"contact not detected (max={contact_max}); 接触检测/prim 路径有问题"
    print("[smoke] PASS")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
