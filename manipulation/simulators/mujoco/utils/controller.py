"""键盘控制器，通过 pynput 监听按键按下/释放，持续控制 SO101 关节。"""
import numpy as np
from pynput import keyboard

# 按键映射：键名 -> (关节索引, 方向)
KEY_MAP = {
    "1": (0, +1), "q": (0, -1),
    "2": (1, +1), "w": (1, -1),
    "3": (2, +1), "e": (2, -1),
    "4": (3, +1), "r": (3, -1),
    "5": (4, +1), "t": (4, -1),
    "6": (5, +1), "y": (5, -1),
}


class KeyboardController:
    """键盘控制器，监听按键按下/释放状态，持续控制 SO101 六个关节。"""

    def __init__(self, model, joint_step=0.02):
        """初始化关节指令数组、步进增量和按键监听。"""
        self.model = model
        self.joint_step = joint_step
        self.q_cmd = np.zeros(model.nu)
        self._pressed = set()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def _on_press(self, key):
        """按键按下回调。"""
        try:
            self._pressed.add(key.char.lower())
        except AttributeError:
            pass

    def _on_release(self, key):
        """按键释放回调。"""
        try:
            self._pressed.discard(key.char.lower())
        except AttributeError:
            pass

    def update(self):
        """查询当前按住的键，持续更新关节指令。"""
        for key, (idx, direction) in KEY_MAP.items():
            if key in self._pressed:
                self.q_cmd[idx] += direction * self.joint_step
                self.q_cmd[idx] = np.clip(self.q_cmd[idx], *self.model.actuator_ctrlrange[idx])

    def get_cmd(self):
        """返回当前关节指令的副本。"""
        return self.q_cmd.copy()

    def reset(self):
        """重置所有关节指令为零。"""
        self.q_cmd[:] = 0
