"""键盘控制器，通过终端 cbreak 模式非阻塞读取按键，控制 SO101 关节。"""
import os
import select
import sys
import termios
import tty

import numpy as np

# 按键映射：字符 -> (关节索引, 方向)
KEY_MAP = {
    "1": (0, +1), "q": (0, -1),
    "2": (1, +1), "w": (1, -1),
    "3": (2, +1), "e": (2, -1),
    "4": (3, +1), "r": (3, -1),
    "5": (4, +1), "t": (4, -1),
    "6": (5, +1), "y": (5, -1),
}


class KeyboardController:
    """键盘控制器，非阻塞读取按键，按住时持续控制关节。"""

    def __init__(self, joint_step=0.02):
        """初始化关节指令数组和终端设置。"""
        self.joint_step = joint_step
        self.model = None
        self.q_cmd = None
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def set_model(self, model):
        """设置 MuJoCo 模型，初始化关节指令数组。"""
        self.model = model
        self.q_cmd = np.zeros(model.nu)

    def update(self):
        """非阻塞读取按键，更新关节指令，返回非关节键列表。"""
        hotkeys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(self._fd, 1).decode("utf-8", errors="ignore")
            if ch in KEY_MAP and self.model is not None:
                idx, direction = KEY_MAP[ch]
                self.q_cmd[idx] += direction * self.joint_step
                self.q_cmd[idx] = np.clip(self.q_cmd[idx], *self.model.actuator_ctrlrange[idx])
            else:
                hotkeys.append(ch)
        return hotkeys

    def get_cmd(self):
        """返回当前关节指令的副本。"""
        return self.q_cmd.copy()

    def reset(self):
        """重置所有关节指令为零。"""
        self.q_cmd[:] = 0

    def restore(self):
        """恢复终端设置。"""
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
