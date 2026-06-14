"""注册 SO101 抓取→放置任务。"""

import gymnasium as gym

from . import agents

gym.register(
    id="ManiBase-SO101-Grasp-Place-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:So101PlaceTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:So101PlacePPORunnerCfg",
    },
)

gym.register(
    id="ManiBase-SO101-Grasp-Place-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:So101PlaceTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:So101PlacePPORunnerCfg",
    },
)

# Student 视觉变体（teacher→student 蒸馏）/ Student vision variant for distillation
gym.register(
    id="ManiBase-SO101-Grasp-Place-Distillation-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_vision_env_cfg:So101PlaceStudentVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:So101PlaceDistillationRunnerCfg",
    },
)

gym.register(
    id="ManiBase-SO101-Grasp-Place-Distillation-Vision-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_vision_env_cfg:So101PlaceStudentVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:So101PlaceDistillationRunnerCfg",
    },
)
