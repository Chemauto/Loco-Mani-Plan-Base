"""注册 SO101 方块抓取任务。"""

import gymnasium as gym

from . import agents

gym.register(
    id="ManiBase-SO101-Grasp-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_grasp_env_cfg:So101GraspCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:So101GraspPPORunnerCfg",
    },
)

gym.register(
    id="ManiBase-SO101-Grasp-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_grasp_env_cfg:So101GraspCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:So101GraspPPORunnerCfg",
    },
)
