"""SO101 方块抓取 PPO 配置。"""

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
class So101GraspPPORunnerCfg:
    class_name = "OnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 2000
    clip_actions = None
    save_interval = 100
    experiment_name = "so101_grasp_cube"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "isaaclab"
    wandb_project = "isaaclab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    torch_compile_mode = None
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    actor = MLPActorCfg()
    critic = MLPCriticCfg()
    algorithm = PPOAlgorithmCfg()
