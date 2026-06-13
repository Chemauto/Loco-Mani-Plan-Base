"""通用数值逆运动学，基于 MuJoCo Jacobian 的阻尼最小二乘法。"""
import numpy as np
import mujoco


class NumericalIK:
    """通用数值 IK，通过迭代 Jacobian 求解任意机器人的逆运动学。"""

    def __init__(self, model, data, max_iter=100, damping=0.01, tolerance=0.001, max_step=0.1):
        """初始化 IK 参数。"""
        self.model = model
        self.data = data
        self.max_iter = max_iter
        self.damping = damping
        self.tolerance = tolerance
        self.max_step = max_step  # 单步关节角增量上限（rad），防止奇异点跳变

    def solve(self, target_pos, ee_body_name="gripper", q_init=None):
        """给定目标位置，迭代求解关节角度。

        target_pos: 目标位置 [x, y, z]
        ee_body_name: 末端执行器 body 名称
        q_init: 初始关节角度，默认使用当前值
        """
        body_id = self.model.body(ee_body_name).id
        q = q_init.copy() if q_init is not None else self.data.qpos[:self.model.nu].copy()

        for _ in range(self.max_iter):
            # 设置关节角度并计算正运动学
            self.data.qpos[:self.model.nu] = q
            mujoco.mj_forward(self.model, self.data)

            # 当前末端位置与误差
            ee_pos = self.data.body(body_id).xpos.copy()
            error = target_pos - ee_pos

            if np.linalg.norm(error) < self.tolerance:
                break

            # 计算 Jacobian（位置部分）
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, None, ee_pos, body_id)

            # 只取可控关节对应的列
            jac = jacp[:, :self.model.nu]

            # 阻尼最小二乘法
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + self.damping ** 2 * np.eye(3), error
            )

            # 步长裁剪：单步增量超过上限时等比例缩放
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_step:
                dq *= self.max_step / dq_abs_max

            q = q + dq
            q = np.clip(q, *self.model.actuator_ctrlrange.T)

        return q
