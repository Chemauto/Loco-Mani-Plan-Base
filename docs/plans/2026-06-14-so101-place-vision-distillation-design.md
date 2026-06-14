# SO101 抓取→放置 + 双相机视觉蒸馏任务设计

> 日期：2026-06-14
> 仓库：`Loco-Mani-Plan-Base` → `manipulation/simulators/isaaclab/ManiBase`
> 风格参考：`/home/xcj/work/IsaacLab/IsaacLabBisShe`（Go2 推箱/行走，`PushBoxTest`）
> 机制参考：`/home/xcj/work/RC2026_SIM/.../arm_control`（SO101 臂抓取/放置 + 视觉蒸馏）
> 基线：现有 `so101_grasp/` 任务（保留不动）

---

## 1. 目标

把现有 `so101_grasp`（纯状态抓取）升级为一个完整的 **抓取→放置 + sim2-real 视觉策略** 任务，采用 **Teacher-Student 蒸馏** 路线：

- **Teacher**：纯状态策略（含特权信息 `ee_to_cube`），RL 训练，收敛快、稳。
- **Student**：图像策略（双相机 + ResNet，**不含特权**），用 Teacher 的动作做 DAgger/BC 蒸馏，最终部署到真机（真机无 ground-truth 物体位姿）。

任务流程：`reach → grab → lift → place`（cube 抓起后送到随机目标点）。

## 2. 风格约束（IsaacLabBisShe）

作为硬约束贯穿实现：

- **MDP 按职责拆独立文件**：`commands.py` / `observations.py` / `rewards.py` / `terminations.py` / `curriculums.py` / `events.py`，`mdp/__init__.py` 用 `from .xxx import *` 统一导出。
- **中英双行注释**：`逻辑 / Logic`、`作用 / Purpose`，配置里用 `###########任务奖励函数###########` 分隔线分类。
- **自定义命令系统**：`CommandTerm` 子类 + `Cfg`，带 `VisualizationMarkers`、`metrics` 诊断、`_resample_command`/`_update_command` 钩子。
- **自定义课程函数**：基于 episode 进度自适应，状态用 `setattr(env, state_name, ...)` 存。
- **特权观测 = 独立 Critic 组**：`enable_corruption=False`，policy 组 `enable_corruption=True` 加噪声；观测用 `@generic_io_descriptor` 装饰。
- **奖励优先 exp 核** `exp(-error/std)`，函数名后缀 `_exp`/`_tanh`；碰撞用 `net_forces_w`，threshold≈1.0。
- **域随机化三模式**：startup(material/mass) + reset(external force/位置) + interval(push)。
- **gym.register 命名**：`Template-<Task>-<Robot>-v0` + `-Play-v0`；Play 变体在 `__post_init__` 关随机化/关噪声/缩 num_envs。

## 3. 目录结构

**新开任务目录** `so101_place/`（与现有 `so101_grasp/` 并列，路径：
`manipulation/simulators/isaaclab/ManiBase/source/ManiBase/ManiBase/tasks/manager_based/so101_place/`）。

```
so101_place/
├── __init__.py                      # gym.register 多个 id
├── so101_place_env_cfg.py           # 基础 EnvCfg（MISSING 占位 robot/cube + 默认奖励/终止/事件/课程）
├── config/
│   ├── __init__.py
│   ├── teacher_env_cfg.py           # Teacher 变体：状态策略，填 MISSING + teacher 观测组
│   ├── student_vision_env_cfg.py    # Student 变体：图像策略，替换观测组为双相机+ResNet
│   └── (各 _PLAY 类在同文件内定义)
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py            # TeacherPPO / StudentDistillation / StudentFinetune runner
└── mdp/
    ├── __init__.py                  # from .commands import * 等，统一导出
    ├── commands.py                  # CubePlaceCommand(CommandTerm) + CubePlaceCommandCfg
    ├── observations.py              # 状态观测 + ResNet10Extractor(ManagerTermBase)
    ├── rewards.py                   # reach/grab/lift/place（exp 核）+ 碰撞惩罚
    ├── terminations.py
    ├── curriculums.py               # place 目标范围自适应 curriculum
    └── events.py                    # 三模式域随机化
```

参数组织采用**混合方案**：结构性内容（用哪些 term、命令系统怎么搭）按 BisShe 风格硬编码在 cfg 类里；每个 term 的**数值**（随机化范围、课程步数、奖励权重、ResNet 维度、相机位姿）走 ManiBase 的 `config.py` + `config.yaml`。

## 4. 场景 + 动作

**`So101PlaceSceneCfg`**（在 `so101_grasp` 基础上扩展，沿用 `SO101_*` 常量与 `SO101_FOLLOWER_CFG`）：

- `robot` / `cube` / `table` / `ee_frame` / `light`：沿用现有配置；cube 加接触求解参数。
- **双相机**（参考 arm_control，位姿/焦距走 yaml）：
  - `jaw_camera`：224×224 RGB，装在夹爪 link（手眼，抓取主视角）。
  - `scene_camera`：224×224 RGB，装在桌面侧（全局视角，看 cube 和目标点）。
- **三个接触传感器**：
  - `jaw_contact_forces`：filter cube，`track_contact_points=True`（夹爪指↔cube）。
  - `gripper_contact_forces`：filter cube（另一指，双指对向判定）。⚠️ 需核对 SO101 MJCF 是否有独立 gripper link；若仅单 jaw 则合并到 jaw。
  - `other_contact_forces`：robot 的 base/arm body，filter 桌面（自碰撞 + 撞桌）。

**`ActionsCfg`**：沿用现有——`arm_action`（6 关节 `JointPositionActionCfg`）+ `gripper_action`（`BinaryJointPositionActionCfg`）。

## 5. 命令系统（核心新增）

`mdp/commands.py`，按 BisShe `BoxGoalCommand` 模式：

```python
class CubePlaceCommand(CommandTerm):
    """在桌面范围内采样 cube 的放置目标位姿 / Sample target pose for placing the cube."""
    cfg: "CubePlaceCommandCfg"
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cube = env.scene[cfg.asset_name]
        self.pos_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
    @property
    def command(self): return self.pos_command_e        # (E,3) 目标点
    def _resample_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_e[env_ids,0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_e[env_ids,1] = r.uniform_(*self.cfg.ranges.pos_y)
        # z 固定在桌面放置高度
    def _update_command(self): pass                      # 静态目标
    def _set_debug_vis_impl(self, debug_vis): ...        # FRAME_MARKER 可视化目标点
```

`CommandsCfg.object_place = mdp.CubePlaceCommandCfg(asset_name="cube", ranges=...)`，目标范围初始小（容易），课程逐步扩大。

## 6. 观测组（4 组 + ResNet + history）

全组 `history_length = 3`。Student policy **不含 `ee_to_cube`**，cube 位置只能从双相机推断——蒸馏的立足点。

**Teacher 变体**：

```python
class PolicyCfg(ObsGroup):        # actor：含特权 ee_to_cube，加噪声
    joint_pos / joint_vel / gripper_pos / ee_to_cube / last_action
    def __post_init__(self): self.history_length=3; self.enable_corruption=True; self.concatenate_terms=True

class CriticCfg(ObsGroup):        # value：更多 ground-truth，不噪声
    # 上述全部 + cube_pose(cube_pose_in_robot_frame) + place_target(command_target) + ee_pos
    def __post_init__(self): self.history_length=3; self.enable_corruption=False
```

**Student 变体**（关键区别：policy 用图像）：

```python
class PolicyCfg(ObsGroup):        # actor：双相机 ResNet 特征，无特权（部署用）
    joint_pos / joint_vel / gripper_pos / last_action
    jaw_img   = ObsTerm(func=mdp.ResNet10Extractor, params={"sensor_cfg": SceneEntityCfg("jaw_camera")})
    scene_img = ObsTerm(func=mdp.ResNet10Extractor, params={"sensor_cfg": SceneEntityCfg("scene_camera")})
    def __post_init__(self): self.history_length=3; self.enable_corruption=True
    # ⚠️ 双 ResNet × 3 帧历史显存较大，跑不动则图像组 history 降到 1

class CriticCfg(ObsGroup):        # value：复用 teacher 特权组（ee_to_cube+cube_pose+target）
    def __post_init__(self): self.history_length=3; self.enable_corruption=False
```

**`ResNet10Extractor`**（`ManagerTermBase` 子类，参考 arm_control）：`__init__` 加载 `helper2424/resnet10`（HF，`trust_remote_code=True`）→ eval → ImageNet 归一化；`__call__` 取 `image(env,...)` 经模型输出 `pooler_output` 特征。观测函数加 `@generic_io_descriptor` 装饰。

**新增观测函数**（`mdp/observations.py`）：`cube_pose_in_robot_frame`（特权完整位姿）、`command_target`（从 `CubePlaceCommand` 读放置目标）；现有 `ee_to_object_vector` / `ee_position_world` 直接复用。

## 7. 奖励（grab→place 全流程，两变体共用）

```python
###############任务奖励（exp 核 + 门控，按 grab→place 顺序）###############
reach_cube  = RewTerm(func=mdp.object_ee_distance_exp,          # exp(-d/std)
                      params={"std":0.10}, weight=_REWARD["reach"])
grab_cube   = RewTerm(func=mdp.gripper_grab_object,             # 双指对向夹持（两指力点积≈-1 + 距离加权）
                      params={"sensor_cfg_jaw":...,"sensor_cfg_gripper":...,"threshold":1.5}, weight=_REWARD["grab"])
lift_cube   = RewTerm(func=mdp.object_lift_height_when_grasped, # 抬升×夹持门控（防泄漏）
                      weight=_REWARD["lift"])
place_track = RewTerm(func=mdp.object_place_distance_exp,       # cube→目标 exp(-d/std)，门控已抬升
                      params={"std":0.10,"command_name":"object_place"}, weight=_REWARD["place"])
place_bonus = RewTerm(func=mdp.place_success_bonus,             # 到目标+稳定 → 一次性奖励
                      params={"command_name":"object_place","dist_threshold":0.03}, weight=_REWARD["success"])

###############惩罚###############
action_rate         = RewTerm(func=mdp.action_rate_l2, weight=_REWARD["action_rate"])
joint_vel           = RewTerm(func=mdp.joint_vel_l2, weight=_REWARD["joint_vel"])
table_coll_jaw      = RewTerm(func=mdp.table_collision, params={"sensor_cfg":...,"threshold":1.0}, weight=-1.0)
table_coll_gripper  = RewTerm(func=mdp.table_collision, ..., weight=-1.0)
self_collision      = RewTerm(func=mdp.self_collision, params={"sensor_cfg":...}, weight=-1.0)
```

新增 reward 函数：`object_ee_distance_exp` / `object_place_distance_exp`（exp 核）、`gripper_grab_object`（arm_control 双指判定）、`object_lift_height_when_grasped`（复用 ManiBase 门控）、`place_success_bonus`、`table_collision` / `self_collision`（复用 arm_control）。权重/std/threshold 走 `config.yaml`。

## 8. 终止

```python
time_out      = DoneTerm(func=mdp.time_out, time_out=True)
place_success = DoneTerm(func=mdp.object_at_target,               # cube 到目标+抬升达标 → 成功
                         params={"command_name":"object_place","dist_threshold":0.03,"height_threshold":0.08})
cube_dropped  = DoneTerm(func=mdp.object_dropped, params={"minimum_height":0.02})  # 掉桌失败
```

沿用 ManiBase 的**相对机器人基座**高度基准（比 arm_control 世界绝对高度更鲁棒）。

## 9. 课程（BisShe `box_goal_progress_curriculum` 风格）

```python
@configclass
class CurriculumCfg:
    place_range = CurrTerm(func=mdp.place_target_progress_curriculum,
                           params={"command_name":"object_place","progress_beta":0.02})
    action_rate = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name":"action_rate","weight":-1e-2,"num_steps":24*600})
```

`place_target_progress_curriculum`（`mdp/curriculums.py`）：`setattr(env, "_place_curriculum_state", {...})` 存基线范围+value；每 episode 算 `progress = 1 - final_dist/initial_dist`，EMA 平滑后**动态扩大** `command_term.cfg.ranges.pos_x/pos_y`——抓得越好目标越远。

## 10. 域随机化（BisShe 三模式）

```python
physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                             params={"static_friction_range":(0.5,1.2),"dynamic_friction_range":(0.5,1.2),"restitution_range":(0.0,0.0),"num_buckets":64})
add_ee_mass      = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup",
                             params={"asset_cfg":SceneEntityCfg("robot",body_names=".*gripper"),"mass_distribution_params":(0.0,0.5),"operation":"add"})
reset_all        = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
reset_cube       = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                             params={"pose_range":{"x":(-0.03,0.03),"y":(-0.05,0.05),"yaw":(-0.4,0.4)},"velocity_range":{},"asset_cfg":SceneEntityCfg("cube")})
```

## 11. 任务变体 + gym.register

`__init__.py`（BisShe 命名 `Template-<Task>-<Robot>-v0`）：

| Task ID | 用途 |
|---|---|
| `Template-Grasp-Place-SO101-v0` | Teacher 训练（状态策略） |
| `Template-Grasp-Place-SO101-Play-v0` | Teacher 回放 |
| `Template-Grasp-Place-SO101-Distillation-Vision-v0` | Student 蒸馏（图像策略） |
| `Template-Grasp-Place-SO101-Distillation-Vision-Play-v0` | Student 回放 |
| `Template-Grasp-Place-SO101-Finetune-Vision-v0` | Student RL 微调（可选） |

## 12. agents

`agents/rsl_rl_ppo_cfg.py`：

- `TeacherPPORunnerCfg`：标准 rsl_rl PPO。
- `StudentDistillationRunnerCfg`：DAgger/BC 蒸馏。⚠️ rsl_rl 无内置蒸馏，需自定义 BC 脚本或 runner（参考 arm_control `lift_distillation_vision_env_cfg`）。
- `StudentFinetunePPORunnerCfg`：蒸馏后 RL 微调。

## 13. config.yaml（混合方案的数值侧）

加到 ManiBase 现有 `config.py` 的 `_DEFAULT_CFG["so101_place"]`，新增段落（全可 yaml override）：

- `scene`（num_envs, env_spacing）
- `camera`（jaw/scene 位姿、rot、焦距、分辨率）
- `resnet`（模型名 `helper2424/resnet10`、特征维度）
- `place`（目标范围 pos_x/pos_y、放置高度、dist_threshold、height_threshold）
- `curriculum`（progress_beta、num_steps）
- `event`（摩擦/质量/扰动范围）
- `reward`（reach/grab/lift/place/success/action_rate/joint_vel 权重 + std + threshold）
- `episode`（length_s, decimation, sim_dt）

## 14. 风险与待办

| 风险/待办 | 处置 |
|---|---|
| SO101 MJCF 是否有独立 gripper link | 实现前核对 `SO101_JAW_LINK_NAME` / 是否有 gripper；单指则合并接触传感器 |
| 双 ResNet × 3 帧历史显存 | 先全组 history=3；图像组 OOM 则降 history=1 |
| 相机位姿/焦距需标定 | 初值参考 arm_control，yaml 可调；手眼相机要确保能看到 cube |
| 蒸馏 runner 非标准 | 先实现 teacher + student 环境与观测，蒸馏脚本单独做 |
| `helper2424/resnet10` 依赖 HF 网络 + `transformers`/`torchvision` | 按 memory 约定 `pip install` 外部依赖，离线则换本地 ResNet |
| 碰撞 filter 误报 | 接触传感器 `filter_prim_paths_expr` 精确指定 cube/table |

## 15. 建议实现顺序

1. **骨架**：`so101_place/` 目录 + `__init__.py` 注册 + 基础 `EnvCfg`（MISSING 占位），先跑通 import。
2. **Teacher 状态版**：场景 + 命令系统 + teacher 观测 + 奖励 + 终止 + 课程 + 随机化，**先不接相机**，训出 teacher 状态策略。
3. **双相机 + ResNet**：加 `jaw_camera`/`scene_camera` + `ResNet10Extractor`，student 观测组。
4. **蒸馏**：student 变体 + 蒸馏 runner/脚本。
5. **调参/Play 变体**：yaml 调权重，`_PLAY` 类关随机化回放。
