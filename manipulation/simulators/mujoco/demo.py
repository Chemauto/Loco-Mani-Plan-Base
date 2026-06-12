"""Load SO101 in MuJoCo and visualize."""
import mujoco
import mujoco.viewer
import yaml


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = mujoco.MjModel.from_xml_path(cfg["robot"]["model_path"])
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
