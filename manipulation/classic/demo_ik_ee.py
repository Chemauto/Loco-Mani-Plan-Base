"""SO101 IK 末端执行器控制 demo。通过 tkinter 面板控制目标位置，IK 求解关节角度。"""
import sys
import tkinter as tk
from pathlib import Path

import mujoco
import numpy as np
import yaml

# 添加 simulators 路径以复用 MujocoSimulator
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulators" / "mujoco"))
from utils.simulator import MujocoSimulator

from ik_analytical import AnalyticalIK
from ik_numerical import NumericalIK


class TargetInput(tk.Frame):
    """目标坐标输入组件：三个输入框 + 确认按钮。"""

    def __init__(self, parent, init_pos, on_confirm, **kwargs):
        """初始化输入框和确认按钮。"""
        super().__init__(parent, **kwargs)
        self.on_confirm = on_confirm
        self.entries = {}

        for j, axis in enumerate("XYZ"):
            tk.Label(self, text=f"{axis}:").grid(row=0, column=j * 2, padx=2)
            ent = tk.Entry(self, width=8)
            ent.grid(row=0, column=j * 2 + 1, padx=2)
            ent.insert(0, f"{init_pos[j]:.4f}")
            self.entries[axis] = ent

        tk.Button(self, text="确认", command=self._apply, width=10).grid(row=1, column=0, columnspan=6, pady=4)

    def _apply(self):
        """读取输入框坐标，调用回调。"""
        try:
            coords = [float(self.entries[a].get()) for a in "XYZ"]
            self.on_confirm(coords)
        except ValueError:
            pass


class IKController:
    """IK 控制面板，管理仿真器、IK 求解器和 tkinter UI。"""

    def __init__(self, config_path="config.yaml"):
        """加载配置，初始化仿真器和 IK 求解器。"""
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.sim = MujocoSimulator(config_path)
        self.model = self.sim.model
        self.data = self.sim.data
        self.ee_body = self.cfg["ik"]["ee_body"]
        self.ik = self._create_ik()
        self.active = False
        self.target_vars = {}

        # 默认姿态（关节归零），在 sim 启动前计算
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.init_pos = self._get_ee_pos()

    def _create_ik(self):
        """根据配置创建 IK 求解器。"""
        if self.cfg["ik"]["method"] == "analytical":
            return AnalyticalIK(self.model, self.data, self.ee_body)
        return NumericalIK(self.model, self.data, **self.cfg["numerical"])

    def _get_ee_pos(self):
        """获取当前末端执行器位置。"""
        return self.data.body(self.model.body(self.ee_body).id).xpos.copy()

    def _on_target_input(self, coords):
        """输入框确认回调，同步坐标到滑条。"""
        self.active = True
        for i, axis in enumerate("XYZ"):
            self.target_vars[axis].set(coords[i])

    def _build_ui(self, root):
        """构建 tkinter 控制面板。"""
        # 滑条
        for i, (axis, val) in enumerate(zip("XYZ", self.init_pos)):
            tk.Label(root, text=f"{axis}:").grid(row=i, column=0, padx=5, pady=8)
            var = tk.DoubleVar(value=round(val, 4))
            tk.Scale(root, variable=var, from_=-0.30, to=0.30, resolution=0.001,
                     orient=tk.HORIZONTAL, length=220,
                     command=lambda _: setattr(self, 'active', True)).grid(row=i, column=1)
            self.target_vars[axis] = var

        # 当前位姿
        self.pos_label = tk.Label(root, text="", font=("Courier", 10))
        self.pos_label.grid(row=3, column=0, columnspan=2, pady=5)

        # 分隔线
        tk.Frame(root, height=2, bg="gray").grid(row=4, column=0, columnspan=2, sticky="ew", padx=10)

        # 坐标输入
        tk.Label(root, text="目标坐标输入:", font=("", 9, "bold")).grid(row=5, column=0, columnspan=2, pady=(8, 2))
        TargetInput(root, self.init_pos, self._on_target_input).grid(row=6, column=0, columnspan=2)

        # 夹爪 + IK 方法
        self.gripper_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="夹爪", variable=self.gripper_var).grid(row=7, column=0, columnspan=2)
        tk.Label(root, text=f"IK: {self.cfg['ik']['method']}", fg="gray").grid(row=8, column=0, columnspan=2)

    def _update(self):
        """主循环：读取目标 → IK 求解 → 控制机器人 → 刷新显示。"""
        if not self.sim.is_running():
            self.root.destroy()
            return

        target = np.array([self.target_vars[a].get() for a in "XYZ"])

        self.sim.locker.acquire()
        ee_pos = self._get_ee_pos()

        # 只在目标与当前位置有偏差时才执行 IK
        if self.active and np.linalg.norm(target - ee_pos) > 1e-3:
            q = self.ik.solve(target, ee_body_name=self.ee_body)
            q[5] = 1.0 if self.gripper_var.get() else 0.0
            self.data.ctrl[:] = q

        self.pos_label.config(text=f"当前: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
        self.sim.locker.release()

        self.root.after(50, self._update)

    def run(self):
        """启动仿真器和控制面板。"""
        self.sim.start()

        self.root = tk.Tk()
        self.root.title("SO101 IK 控制")
        self.root.geometry("380x380")
        self.root.resizable(False, False)

        self._build_ui(self.root)
        self._update()
        self.root.mainloop()

        self.sim.wait()


if __name__ == "__main__":
    IKController().run()
