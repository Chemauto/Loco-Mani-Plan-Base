"""SO101 抓取→放置任务：课程学习。

逻辑 / Logic:
    根据 episode 内放置目标的达成进度，自适应扩大目标随机范围。
    抓/放得越好（progress 越高），目标范围越大，逐步增加难度。
    状态用 setattr(env, ...) 挂在环境上（贴 BisShe box_goal_progress_curriculum 风格）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def place_target_progress_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "object_place",
    progress_beta: float = 0.02,
) -> float:
    """按放置达成进度自适应扩大目标范围 / Expand target range by placement progress.

    逻辑 / Logic:
        progress = clamp(1 - final_error / initial_error, 0, 1)
        value <- (1 - beta) * value + beta * mean(progress)
        目标范围 <- base_range * (1 + value)（value 越大范围越大）
    """
    command_term = env.command_manager.get_term(command_name)
    state_name = "_place_curriculum_state"
    if not hasattr(env, state_name):
        setattr(
            env,
            state_name,
            {
                "base_pos_x": tuple(command_term.cfg.ranges.pos_x),
                "base_pos_y": tuple(command_term.cfg.ranges.pos_y),
                "value": 0.0,
            },
        )
    state = getattr(env, state_name)

    # episode 进度：初始误差 -> 当前误差的改善比例 / improvement from initial to current error
    initial = command_term.initial_error_pos[env_ids]
    final = command_term.metrics["error_pos"][env_ids]
    progress = torch.clamp(1.0 - final / (initial + 1e-6), min=0.0, max=1.0)
    progress_mean = progress.mean()

    # EMA 平滑 / EMA smoothing
    state["value"] = (1.0 - progress_beta) * state["value"] + progress_beta * progress_mean.item()

    # value 越高 -> 目标范围越大 / higher value -> larger target range
    scale = 1.0 + state["value"]
    base_x, base_y = state["base_pos_x"], state["base_pos_y"]
    cx = (base_x[0] + base_x[1]) / 2
    hx = (base_x[1] - base_x[0]) / 2 * scale
    cy = (base_y[0] + base_y[1]) / 2
    hy = (base_y[1] - base_y[0]) / 2 * scale
    command_term.cfg.ranges.pos_x = (cx - hx, cx + hx)
    command_term.cfg.ranges.pos_y = (cy - hy, cy + hy)

    return state["value"]
