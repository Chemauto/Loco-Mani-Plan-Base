"""SO101 解析逆运动学，从 MuJoCo 模型自动提取几何参数。"""
import math
import mujoco
import numpy as np


class AnalyticalIK:
    """SO101 专用解析 IK，从模型自动计算连杆参数和关节偏移。"""

    def __init__(self, model, data, ee_body="gripper"):
        """从 MuJoCo 模型提取几何参数。"""
        # 保存仿真状态，用零位计算几何参数
        qpos_bak = data.qpos.copy()
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)

        shoulder = data.body(model.body("shoulder").id).xpos.copy()
        upper_arm = data.body(model.body("upper_arm").id).xpos.copy()
        lower_arm = data.body(model.body("lower_arm").id).xpos.copy()
        wrist = data.body(model.body("wrist").id).xpos.copy()
        ee = data.body(model.body(ee_body).id).xpos.copy()

        # 肩关节世界坐标
        self.shoulder_pos = shoulder

        # shoulder_lift 相对肩关节的臂平面偏移
        rel_base = upper_arm - shoulder
        self.r_base = math.sqrt(rel_base[0] ** 2 + rel_base[1] ** 2)
        self.h_base = rel_base[2]

        # 连杆长度
        self.l1 = np.linalg.norm(lower_arm - upper_arm)
        self.l2 = np.linalg.norm(wrist - lower_arm)

        # 腕到末端的水平偏移
        w2e = ee - wrist
        self.d_wrist = math.sqrt(w2e[0] ** 2 + w2e[1] ** 2)

        # 用 q=0 的腕位置反解 theta 偏移
        rel_wc = wrist - upper_arm
        r_wc = math.sqrt(rel_wc[0] ** 2 + rel_wc[1] ** 2)
        h_wc = rel_wc[2]
        d = math.sqrt(r_wc ** 2 + h_wc ** 2)
        cos_t2 = np.clip(-(d ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1, 1)
        theta2 = math.pi - math.acos(cos_t2)
        theta1 = math.atan2(h_wc, r_wc) + math.atan2(
            self.l2 * math.sin(theta2), self.l1 + self.l2 * math.cos(theta2)
        )
        self.theta1_offset = theta1
        self.theta2_offset = theta2

        # 恢复仿真状态
        data.qpos[:] = qpos_bak
        mujoco.mj_forward(model, data)

    def solve(self, target_pos, ee_body_name=None):
        """给定目标位置 [x, y, z]，返回 6 个关节角度。"""
        # 相对肩关节
        dx = target_pos[0] - self.shoulder_pos[0]
        dy = target_pos[1] - self.shoulder_pos[1]
        dz = target_pos[2] - self.shoulder_pos[2]

        # shoulder_pan（轴为 -Z，取反）
        q0 = -math.atan2(dy, dx)

        # 臂平面坐标：减去 shoulder_lift 偏移和腕偏移，得到腕中心相对 shoulder_lift
        r = math.sqrt(dx ** 2 + dy ** 2) - self.r_base - self.d_wrist
        h = dz - self.h_base

        # 垂直面双连杆（余弦定理）
        d = math.sqrt(r ** 2 + h ** 2)
        d = self._clamp_reach(d)
        scale = d / math.sqrt(r ** 2 + h ** 2 + 1e-10)
        r *= scale
        h *= scale

        cos_t2 = np.clip(-(d ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1, 1)
        theta2 = math.pi - math.acos(cos_t2)
        theta1 = math.atan2(h, r) + math.atan2(
            self.l2 * math.sin(theta2), self.l1 + self.l2 * math.cos(theta2)
        )

        # 关节映射（q1 取反号，q2 正号）
        q1 = self.theta1_offset - theta1
        q2 = theta2 - self.theta2_offset

        # 末端姿态（补偿前臂角度，保持夹爪水平）
        q3 = q1 + q2
        q4 = 0.0
        q5 = 0.0

        return np.array([q0, q1, q2, q3, q4, q5])

    def _clamp_reach(self, d):
        """将距离限制在可达范围内。"""
        r_max = self.l1 + self.l2
        r_min = abs(self.l1 - self.l2)
        return np.clip(d, r_min + 1e-4, r_max - 1e-4)
