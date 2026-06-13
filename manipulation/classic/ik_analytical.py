"""SO101 解析逆运动学，基于余弦定理拆解为三个独立子问题。"""
import math
import numpy as np


class AnalyticalIK:
    """SO101 专用解析 IK，将 5DOF 问题拆解为水平方向 + 垂直双连杆 + 末端姿态。"""

    def __init__(self, l1=0.1159, l2=0.1350):
        """初始化连杆长度和关节偏移。"""
        self.l1 = l1
        self.l2 = l2
        # 垂直平面内的关节角度偏移（补偿 DH 参数与实际关节角的差异）
        self.theta1_offset = -math.atan2(0.028, 0.11257)
        self.theta2_offset = -math.atan2(0.0052, 0.1349) + self.theta1_offset

    def solve(self, target_pos):
        """给定目标位置 [x, y, z]，返回 6 个关节角度。如果不可达则缩放到最近可达点。"""
        x, y, z = target_pos

        # 子问题 1：水平面方向
        q0 = math.atan2(y, x)

        # 投影到垂直面
        r = math.sqrt(x ** 2 + y ** 2)
        h = z

        # 子问题 2：垂直面双连杆（余弦定理）
        d = math.sqrt(r ** 2 + h ** 2)
        d = self._clamp_reach(d)
        r_scaled, h_scaled = r * d / math.sqrt(r ** 2 + h ** 2 + 1e-10), h * d / math.sqrt(r ** 2 + h ** 2 + 1e-10)

        cos_theta2 = -(d ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)
        theta2 = math.pi - math.acos(cos_theta2)
        theta1 = math.atan2(h_scaled, r_scaled) + math.atan2(
            self.l2 * math.sin(theta2), self.l1 + self.l2 * math.cos(theta2)
        )

        q1 = theta1 - self.theta1_offset
        q2 = theta2 - self.theta2_offset

        # 子问题 3：末端姿态（SO101 为 5DOF，yaw 不可控）
        q3 = -theta1
        q4 = 0.0
        q5 = 0.0

        return np.array([q0, q1, q2, q3, q4, q5])

    def _clamp_reach(self, d):
        """将距离限制在可达范围内。"""
        r_max = self.l1 + self.l2
        r_min = abs(self.l1 - self.l2)
        return np.clip(d, r_min + 1e-4, r_max - 1e-4)
