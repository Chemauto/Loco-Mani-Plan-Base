"""SO101 抓取→放置 PPO 配置。

注意 / Note:
    obs_groups 采用非对称 actor-critic：actor 看 policy 组，critic 额外看 critic 特权组。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
)


@configclass
class MLPActorCfg:
    class_name = "MLPModel"
    hidden_dims = [256, 128, 64]
    activation = "elu"
    obs_normalization = True
    distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
    }


@configclass
class MLPCriticCfg:
    class_name = "MLPModel"
    hidden_dims = [256, 128, 64]
    activation = "elu"
    obs_normalization = True
    distribution_cfg = None


@configclass
class PPOAlgorithmCfg:
    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    normalize_advantage_per_mini_batch = False
    rnd_cfg = None
    symmetry_cfg = None


@configclass
class So101PlacePPORunnerCfg:
    """SO101 抓取→放置 Teacher PPO runner 配置。"""

    class_name = "OnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 3000
    clip_actions = None
    save_interval = 100
    experiment_name = "so101_place"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "isaaclab"
    wandb_project = "isaaclab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    torch_compile_mode = None
    # 非对称 AC：critic 额外看特权组 / asymmetric AC: critic also sees privileged group
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }
    actor = MLPActorCfg()
    critic = MLPCriticCfg()
    algorithm = PPOAlgorithmCfg()


#########################
# Student Distillation ##  teacher→student 蒸馏
#########################


@configclass
class So101PlaceDistillationRunnerCfg(So101PlacePPORunnerCfg):
    """SO101 抓取→放置 teacher→student 蒸馏配置。

    逻辑 / Logic:
        teacher（状态策略，看 teacher 观测组）实时推理，student（图像策略，看 policy 观测组）
        用 MSE 行为克隆模仿 teacher 动作。teacher checkpoint 自动从 logs/rsl_rl/so101_place/ 加载。
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    class_name = "DistillationRunner"
    # 同 teacher 的 experiment_name，以便 train.py 的 get_checkpoint_path 自动找到 teacher model
    experiment_name = "so101_place"
    run_name = "distillation"

    obs_groups = {
        "policy": ["policy"],     # student（本体感觉 + 双 ResNet 特征，无 ee_to_cube）
        "teacher": ["teacher"],   # teacher（本体感觉 + ee_to_cube 特权）
        "critic": ["critic"],     # critic（ground-truth 特权）
    }

    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[256, 128, 64],
        teacher_obs_normalization=True,
        student_obs_normalization=True,
        activation="elu",
        init_noise_std=0.1,
        class_name="StudentTeacherRecurrent",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
        teacher_recurrent=False,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-3,
        gradient_length=5 * (2048 / 16),
        optimizer="adam",
        loss_type="mse",
    )
