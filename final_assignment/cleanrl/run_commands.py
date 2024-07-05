import os
import pathlib
import subprocess

wandb_project_name = "finalAssignmentResit"

########################################################################################################################
# Testing to see if there is a difference between one and ten environments in terms of model performance
########################################################################################################################
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "100_000",
        # "50_000",
        "--num-envs",
        "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
    ],
)

subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "100_000",
        # "50_000",
        # "--num-envs",
        # "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
    ],
)

########################################################################################################################
# Testing DDPG for speed
########################################################################################################################
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ddpg_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        # "1_000_000",
        "50_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
    ],
)

########################################################################################################################
# Real test
########################################################################################################################

# -------------------------------------------------------------------------------------------------------------------- #
# PPO
# -------------------------------------------------------------------------------------------------------------------- #
# 2 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        # "50_000",
        "--num-envs",
        "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# 4 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        # "50_000",
        "--num-envs",
        "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# 6 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        # "50_000",
        "--num-envs",
        "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint_ppo = checkpoint_models[-1]

# -------------------------------------------------------------------------------------------------------------------- #
# DDPG
# -------------------------------------------------------------------------------------------------------------------- #
# Run DDPG now continue with PPO later.
# 100 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ddpg_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "1_000_000",
        # "50_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# 2 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ddpg_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "1_000_000",
        # "50_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint_ddpg = checkpoint_models[-1]

# Continue with PPO
# 8 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        # "50_000",
        "--num-envs",
        "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint_ppo),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# 10 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        # "50_000",
        "--num-envs",
        "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# 12 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        # "50_000",
        "--num-envs",
        "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

# Continue with DDPG
# 3 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ddpg_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "1_000_000",
        # "50_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint_ddpg),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# 4 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ddpg_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "1_000_000",
        # "50_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint_ddpg),
    ],
)

# Run PPO for another 8 000 000 timesteps
latest_checkpoint = "./runs/Pendulum-v1__ppo_continuous_action__1__1720159844/ppo_continuous_action_checkpoint_1986560.cleanrl_model"
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "8_000_000",
        # "50_000",
        "--num-envs",
        "10",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

# Run PPO with only 1 environment for 10 000 000 timesteps
# First 2 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# Next 2 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# Next 2 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# Next 2 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]

# Last 2 000 000 timesteps
subprocess.run(
    [
        "poetry",
        "run",
        "python",
        "cleanrl/ppo_continuous_action.py",
        "--env-id",
        "Pendulum-v1",
        "--total-timesteps",
        "2_000_000",
        "--track",
        "--wandb-project-name",
        wandb_project_name,
        "--capture-video",
        "--save-model",
        "--save-checkpoints",
        "--checkpoint-file",
        str(latest_checkpoint),
    ],
)

created_run_dir = max(pathlib.Path("./runs").glob("*/"), key=os.path.getmtime)
checkpoint_models = sorted(
    created_run_dir.glob("*_checkpoint_*"),
    key=lambda x: int(x.stem.rsplit("_", 1)[-1]),
)
latest_checkpoint = checkpoint_models[-1]
