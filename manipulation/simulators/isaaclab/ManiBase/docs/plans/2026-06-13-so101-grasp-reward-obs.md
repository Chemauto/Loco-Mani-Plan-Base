# SO101 抓取任务 — 补全 grasp/contact 奖励与观测 修复训练不收敛

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 SO101 抓取 PPO 训练能真正学会抓起方块（当前 2000 iter success≈0，episode 全超时）。

**Architecture:** 不引入视觉，先把低维状态任务做对。根因是奖励在 "末端靠近方块"→"方块被抬起" 之间没有密集信号，且观测缺少夹爪状态/末端位置、lift 奖励有泄漏。改动分两批：P0 补 contact-based grasp 奖励 + 夹爪观测 + 修复 lift 泄漏；P1 补末端观测并修掉 `cube_to_ee` 命名 bug。所有新增权重/阈值进 `config.yaml`。

**Tech Stack:** IsaacLab manager-based RL，PyTorch，RSL-RL PPO，SO101 URDF（单活动夹爪 `moving_jaw_so101_v1_link`）。

---

## 关键设计决策（与参考项目的差异）

1. **SO101 是单活动夹爪**，不是 arm_control 的双 jaw。所以 grasp 检测用 IsaacLab `ContactSensorCfg.filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"]`，只上报 `moving_jaw_so101_v1_link` 与 Cube 的接触——天然排除桌面误报，比 arm_control 的对向力检测更简单可靠。
2. **lift 泄漏修复**用乘法门控（参考 arm_control 的 `object_ee_distance_and_lifted`）：`lift = lift_height * is_grasping_contact`。只有夹住才给抬升奖励。
3. **`cube_to_ee` 是命名 bug**（实际算的是 cube 相对 robot root），P1 改成真正的 ee→cube 向量并重命名。

---

## Task 1 [P0]: 场景加夹爪-Cube 接触传感器

**Files:**
- Modify: `source/ManiBase/ManiBase/tasks/manager_based/so101_grasp/so101_grasp_env_cfg.py`（`So101GraspSceneCfg`）

**改动：** 在 `ee_frame` 之后加：

```python
from isaaclab.sensors import ContactSensorCfg  # 顶部 import 区

jaw_contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    history_length=0,
    track_air_time=False,
)
```

**验证：** 见 Task 5 冒烟测试（传感器能在 finger 接近 cube 时上报非零力）。

---

## Task 2 [P0]: 写 grasp 奖励 + 门控 lift 奖励

**Files:**
- Modify: `source/ManiBase/ManiBase/tasks/manager_based/so101_grasp/mdp/rewards.py`

**改动：** 新增两个函数，保留旧函数兼容：

```python
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def gripper_object_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """夹爪活动指与目标物体的接触（已用 filter 只报 finger<->cube）。>阈值记 1。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.linalg.vector_norm(force, dim=-1) > threshold
    return torch.any(contact, dim=-1).float()


def object_lift_height_when_grasped(
    env,
    object_cfg,
    robot_cfg,
    sensor_cfg,
    robot_base_name: str = "base_link",
    threshold: float = 1.0,
) -> torch.Tensor:
    """抬升高度，但仅在夹住物体时计奖（修复泄漏）。"""
    height = object_lift_height(env, object_cfg, robot_cfg, robot_base_name)
    grasping = gripper_object_contact(env, sensor_cfg, threshold)
    return height * grasping
```

---

## Task 3 [P0]: 观测加夹爪关节位置

**Files:**
- Modify: `so101_grasp_env_cfg.py` 的 `ObservationsCfg.PolicyCfg`

**改动：** 加一项读夹爪关节位置：

```python
gripper_pos = ObsTerm(
    func=mdp.joint_pos_rel,
    params={"joint_ids": [SO101_GRIPPER_JOINT_NAME]},  # 见实现：用 joint_names 索引
)
```
（实现时确认 `joint_pos_rel` 接受 `joint_names`/`joint_ids`；若不接，写一个本地 obs 包装。）

---

## Task 4 [P0]: 接线 reward / scene / config.yaml

**Files:**
- Modify: `so101_grasp_env_cfg.py` 的 `RewardsCfg`
- Modify: `config.py` 的 `_DEFAULT_CFG` 和字段映射
- Modify: `config.yaml`

**改动：**
- `lift_cube` 改用 `object_lift_height_when_grasped`，加 `sensor_cfg=SceneEntityCfg("jaw_contact_forces")`。
- 新增 `grab_cube = RewTerm(func=mdp.gripper_object_contact, weight=grasp_weight, params={sensor_cfg, threshold})`。
- `config.yaml` 加 `grasp_weight: 6.0`、`contact_force_threshold: 1.0`。
- `config.py._DEFAULT_CFG["reward"]` 同步加这两键。

---

## Task 5 [P1]: 观测补末端 + 修 `cube_to_ee` 命名

**Files:**
- Modify: `mdp/observations.py`
- Modify: `so101_grasp_env_cfg.py` 的 `ObservationsCfg.PolicyCfg`

**改动：**
- `observations.py` 新增真正的 ee→cube 向量：

```python
def ee_to_object_vector(env, ee_frame_cfg, object_cfg, ee_frame_index: int = 0) -> torch.Tensor:
    """末端指向物体的向量（世界系）。"""
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :]
    return obj.data.root_pos_w[:, :3] - ee_pos
```

- `PolicyCfg` 把 `cube_to_ee`（误导命名）替换为 `ee_to_cube = ObsTerm(func=mdp.ee_to_object_vector, ...)`。
- 加 `ee_pos`（可选）：直接给末端世界坐标。

---

## Task 6: 冒烟测试（替代单元测试，RL 奖励需真实仿真）

**Files:**
- Create: `source/ManiBase/ManiBase/tasks/manager_based/so101_grasp/tests/test_smoke.py`（或 scripts/）

**做法：** headless 起 `num_envs=4` 的环境，跑 ~60 步随机动作，断言：
1. 不报错、不抛 link-not-found。
2. 每个 obs term 形状符合预期，policy 总观测维度 = 各项之和。
3. 每个 reward term 全 finite；`grab_cube` 在 finger 碰到 cube 时 >0；`lift_cube` 在未抓取时为 0（验证泄漏修复）。
4. ContactSensor `net_forces_w` 在接触时非零。

**运行：**
```bash
/home/xcj/miniconda3/envs/env_isaaclab/bin/python -m pytest <test> -v
# 或直接跑脚本 --headless
```

---

## Task 7: 提交

```bash
git add ...
git commit -m "feat(so101_grasp): add grasp/contact reward, gripper obs, fix lift leakage & cube_to_ee naming"
```

---

## 不做（YAGNI）

- 视觉/相机（等状态任务 success 起来再做，且不用 trust_remote_code 的 ResNet10）。
- curriculum、collision 惩罚、reach 改 tanh —— 收敛后再调。
- teacher-student distillation。
