<div align="center">

# 🤖 Loco-Mani-Plan-Base

**统一的机器人学习入门框架**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![IsaacLab](https://img.shields.io/badge/Simulator-IsaacLab-orange.svg)](https://isaac-sim.github.io/IsaacLab/)
[![LeRobot](https://img.shields.io/badge/Policy-LeRobot-purple.svg)](https://github.com/huggingface/lerobot)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-red.svg)](https://github.com/google-deepmind/mujoco)

[English](README.md) | [简体中文](#)

</div>

---

## 🔥 为什么需要 Loco-Mani-Plan-Base？

机器人技术正以前所未有的速度发展，然而学习曲线依然陡峭。
运动控制、操作与规划分散在不同的代码库、互不兼容的环境和零散的教程中。

**Loco-Mani-Plan-Base** 旨在弥合这一鸿沟：提供一个统一的、精心整理的入门起点，
将前沿工具整合为连贯且对新手友好的学习体验。

> 🎯 从零开始，一个仓库跑通机器人。

## ✨ 核心特性

- 🦿 **运动控制 (Locomotion)** — 基于强化学习的足式与轮式运动控制
- 🦾 **操作 (Manipulation)** — 灵巧操作与双臂协作任务
- 🧠 **规划 (Planner)** — 运动规划与任务级推理
- 📦 **开箱即用** — 最小化配置，即刻开始实验
- 🎓 **教程驱动** — 每个模块均配备分步教学 notebook
- 🔧 **模块化设计** — 各模块可独立使用，也可组合运行

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/your-username/Loco-Mani-Plan-Base.git
cd Loco-Mani-Plan-Base

# 创建 conda 环境
conda create -n lmp python=3.11 -y
conda activate lmp

# 安装依赖
pip install -r requirements.txt
```

## 📂 仓库结构

```
Loco-Mani-Plan-Base/
├── locomotion/      # 🦿 运动控制策略与训练
├── manipulation/    # 🦾 操作技能与数据集
├── planner/         # 🧠 运动规划与任务推理
└── README.md
```

## 🙏 致谢

本项目基于机器人社区的杰出工作构建：

- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** — GPU 加速的机器人仿真平台
- **[LeRobot](https://github.com/huggingface/lerobot)** — 让真实世界机器人学习触手可及
- **[MuJoCo](https://github.com/google-deepmind/mujoco)** — 高保真物理仿真引擎

## 📜 开源协议

本项目采用 [Apache License 2.0](LICENSE) 开源协议 — 与 Isaac Lab、LeRobot 及 MuJoCo 保持一致。

---

<div align="center">

**[⬆ 回到顶部](#-loco-mani-plan-base)**

如果这个项目对你有帮助，请考虑给个 ⭐️！

</div>
