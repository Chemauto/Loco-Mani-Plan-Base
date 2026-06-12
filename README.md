<div align="center">

# 🤖 Loco-Mani-Plan-Base

**A Unified Beginner-Friendly Framework for Robot Learning**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![IsaacLab](https://img.shields.io/badge/Simulator-IsaacLab-orange.svg)](https://isaac-sim.github.io/IsaacLab/)
[![LeRobot](https://img.shields.io/badge/Policy-LeRobot-purple.svg)](https://github.com/huggingface/lerobot)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-red.svg)](https://github.com/google-deepmind/mujoco)

[English](#) | [简体中文](README_zh.md)

</div>

---

## 🔥 Why Loco-Mani-Plan-Base?

Robotics is evolving at an unprecedented pace — yet the learning curve remains daunting.
Locomotion, manipulation, and planning are scattered across disconnected codebases,
incompatible environments, and fragmented tutorials.

**Loco-Mani-Plan-Base** bridges this gap: a single, curated entry point that integrates
state-of-the-art tools into a cohesive, beginner-friendly experience.

> 🎯 From zero to a running robot — one repository.

## ✨ Key Features

- 🦿 **Locomotion** — Legged & wheeled locomotion with RL training
- 🦾 **Manipulation** — Dexterous manipulation & bimanual tasks
- 🧠 **Planner** — Motion planning & task-level reasoning
- 📦 **Plug-and-play** — Minimal setup, start experimenting immediately
- 🎓 **Tutorial-driven** — Every module comes with step-by-step notebooks
- 🔧 **Modular design** — Use each module independently or together

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/Loco-Mani-Plan-Base.git
cd Loco-Mani-Plan-Base

# Create conda environment
conda create -n lmp python=3.11 -y
conda activate lmp

# Install dependencies
pip install -e .
```

## 📂 Repository Structure

```
Loco-Mani-Plan-Base/
├── locomotion/      # 🦿 Locomotion policies & training
├── manipulation/    # 🦾 Manipulation skills & datasets
├── planner/         # 🧠 Motion planning & task reasoning
└── README.md
```

## 🙏 Acknowledgements

This project builds upon the incredible work of the robotics community:

- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** — GPU-accelerated robot simulation
- **[LeRobot](https://github.com/huggingface/lerobot)** — Real-world robot learning made accessible
- **[MuJoCo](https://github.com/google-deepmind/mujoco)** — High-fidelity physics engine

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE) — the same permissive license used by Isaac Lab, LeRobot, and MuJoCo.

---

<div align="center">

**[⬆ Back to Top](#-loco-mani-plan-base)**

If you find this project helpful, please consider giving it a ⭐️!

</div>
