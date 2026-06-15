"""SO101 抓取→放置 PPO 配置。

注意 / Note:
    obs_groups 采用非对称 actor-critic：actor 看 policy 组，critic 额外看 critic 特权组。
"""

from isaaclab.utils import configclass


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


@configclass
class StudentModelCfg:
    """student 网络配置（MLP，处理图像特征观测）/ Student network config."""

    class_name = "MLPModel"
    hidden_dims = [512, 256, 128]
    activation = "elu"
    obs_normalization = True
    distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 0.1,
        "std_type": "scalar",
    }


@configclass
class TeacherModelCfg:
    """teacher 网络配置（MLP，处理状态观测）/ Teacher network config.

    注意 / Note: distribution_cfg 必须与 teacher 训练时的 MLPActorCfg 一致
    （init_std=0.8），否则 load_state_dict 会因 distribution.std_param key 不匹配而失败。
    """

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
class DistillationAlgorithmCfg:
    """蒸馏算法配置 / Distillation algorithm config."""

    class_name = "Distillation"
    num_learning_epochs = 5
    learning_rate = 1e-3
    gradient_length = 5 * (2048 / 16)
    optimizer = "adam"
    loss_type = "mse"


#########################
# Student Distillation ##  teacher→student 蒸馏
#########################


@configclass
class So101PlaceDistillationRunnerCfg(So101PlacePPORunnerCfg):
    """SO101 抓取→放置 teacher→student 蒸馏配置。

    逻辑 / Logic:
        teacher（状态策略，看 teacher 观测组）实时推理，student（图像策略，看 student 观测组）
        用 MSE 行为克隆模仿 teacher 动作。teacher checkpoint 自动从 logs/rsl_rl/so101_place/ 加载。

    注意 / Note:
        my_rsl_rl 的 DistillationRunner.construct_algorithm 期望 cfg 有独立的 "student" 和 "teacher"
        顶层 dict（各含 class_name），obs_groups 需有 "student" 和 "teacher" key。
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    class_name = "DistillationRunner"
    # 同 teacher 的 experiment_name，以便 train.py 的 get_checkpoint_path 自动找到 teacher model
    experiment_name = "so101_place"
    run_name = "distillation"

    # my_rsl_rl DistillationRunner 期望 obs_groups 有 "student" 和 "teacher" key
    obs_groups = {
        "student": ["policy"],    # student 看 policy 观测组（图像+本体感觉）
        "teacher": ["teacher"],   # teacher 看 teacher 观测组（状态+特权）
    }

    student = StudentModelCfg()
    teacher = TeacherModelCfg()
    algorithm = DistillationAlgorithmCfg()
