# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import pathlib
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.experimental.wrappers.rendering import RecordVideoV0 as RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Custom arguments
    save_checkpoints: bool = False
    """if toggled, save model checkpoints every iteration"""
    checkpoint_file: str = None
    """the path to the checkpoint file to use to train further"""


def make_env(env_id, seed, idx, capture_video, run_name, gravity: float | None = None):
    def thunk():
        if capture_video and idx == 0:
            # -------------------------------------------------------------------------------------------------------- #
            # Changed by Bas
            # -------------------------------------------------------------------------------------------------------- #
            # To test if the policy can handle different gravity settings
            if gravity is not None:
                env = gym.make(env_id, render_mode="rgb_array", g=gravity)
            else:
                env = gym.make(env_id, render_mode="rgb_array")

            # -------------------------------------------------------------------------------------------------------- #
            # Changed by Bas
            # -------------------------------------------------------------------------------------------------------- #
            # Does not work with mujoco after the first video the other videos are empty,
            # however then the videos are not send to wandb
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            if gravity is not None:
                # Only sending training videos to wandb
                env = RecordVideo(env, f"videos/{run_name}_gravity_{gravity}")
                # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}_gravity_{gravity}")
            else:
                # env = RecordVideo(env, f"videos/{run_name}")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if gravity is not None:
                env = gym.make(env_id, render_mode="rgb_array", g=gravity)
            else:
                env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)

    if args.checkpoint_file:
        checkpoint_file = pathlib.Path(args.checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        qf1.load_state_dict(checkpoint['qf1_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        print(f"checkpoint loaded from {args.checkpoint_file}")

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())


    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # -------------------------------------------------------------------------------------------------------- #
                # Changed by Bas
                # -------------------------------------------------------------------------------------------------------- #
                writer.add_scalar("charts/mean_action", actions.mean().item(), global_step)
                writer.add_scalar("charts/action_std", actions.std().item(), global_step)
                # Add histogram to tensorboard and wandb
                actions_flattened = actions.flatten()
                writer.add_histogram("charts/action_distribution", actions_flattened, global_step, bins=512)

            # -------------------------------------------------------------------------------------------------------- #
            # Changed by Bas
            # -------------------------------------------------------------------------------------------------------- #
            # Save a checkpoint every iteration.
            if args.save_checkpoints and global_step % 5_000 == 0:
                model_checkpoint_path = f"runs/{run_name}/{args.exp_name}_checkpoint_{global_step}.cleanrl_model"
                torch.save({
                    'actor_state_dict': actor.state_dict(),
                    'qf1_state_dict': qf1.state_dict(),
                    'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                    'q_optimizer_state_dict': q_optimizer.state_dict(),
                }, model_checkpoint_path)
                print(f"model checkpoint saved to {model_checkpoint_path}")

                # Save model to wandb
                if args.track:
                    artifact = wandb.Artifact(f"{args.exp_name}_checkpoint", type="model", metadata=dict(
                        global_step=global_step,
                    ))
                    artifact.add_file(model_checkpoint_path)
                    wandb.run.log_artifact(artifact)

    # Log runtime
    writer.add_scalar("charts/train_time", time.time() - start_time)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)

        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ddpg_eval import evaluate

        # -------------------------------------------------------------------------------------------------------- #
        # Changed by Bas
        # -------------------------------------------------------------------------------------------------------- #
        # Save model to wandb
        if args.track:
            artifact = wandb.Artifact(f"{args.exp_name}_model", type="model")
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)
        # -------------------------------------------------------------------------------------------------------- #
        # Changed by Bas
        # -------------------------------------------------------------------------------------------------------- #
        # Generate linear distribution of gravity settings
        gravities = np.linspace(0, 30, 60).round(2)
        gravities = np.concatenate(([None], gravities))
        episodic_returns_list = []
        mean_gravity_returns = []
        std_gravity_returns = []
        min_gravity_returns = []
        max_gravity_returns = []

        for gravity in gravities:
            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=(Actor, QNetwork),
                device=device,
                exploration_noise=args.exploration_noise,
                gravity=gravity,
            )
            # for idx, episodic_return in enumerate(episodic_returns):
            #     writer.add_scalar(f"eval/episodic_return_gravity_{gravity}", episodic_return, idx)

            # -------------------------------------------------------------------------------------------------------- #
            # Changed by Bas
            # -------------------------------------------------------------------------------------------------------- #
            # Save episodic returns
            episodic_returns_list.append(np.concatenate(episodic_returns, axis=0).tolist())
            mean_gravity_returns.append(np.mean(episodic_returns))
            std_gravity_returns.append(np.std(episodic_returns))
            min_gravity_returns.append(np.min(episodic_returns))
            max_gravity_returns.append(np.max(episodic_returns))
            # Save videos to wandb
            videos_dir = pathlib.Path(__file__).resolve().parents[1] / "videos"
            created_videos_dir = max(videos_dir.glob("*/"), key=os.path.getmtime)
            videos = list(created_videos_dir.glob("*.mp4"))
            for video in videos:
                wandb.log({f"videos_{gravity}": wandb.Video(str(video))})

        gravity_keys = np.linspace(0, 30, 60).round(2)
        gravity_keys = [f'gravity_{key}' for key in gravity_keys]
        gravity_keys = np.concatenate((["default"], gravity_keys))

        # -------------------------------------------------------------------------------------------------------- #
        # Changed by Bas
        # -------------------------------------------------------------------------------------------------------- #
        # Log episodic returns to wand
        wandb.log({
            "eval/episodic_return": wandb.plot.line_series(
                xs=np.arange(len(episodic_returns)),
                ys=episodic_returns_list,
                keys=gravity_keys,
                title="Episodic Returns",
            ),
        })

        data = [[x, y] for (x, y) in zip(gravity_keys, mean_gravity_returns)]
        table = wandb.Table(data=data, columns=["gravity", "mean_episodic_return"])
        wandb.log({"eval/mean_episodic_return": wandb.plot.bar(table, "gravity", "mean_episodic_return",
                                                               title="Mean Episodic Return")})

        data = [[x, y] for (x, y) in zip(gravity_keys, std_gravity_returns)]
        table = wandb.Table(data=data, columns=["gravity", "std_episodic_return"])
        wandb.log({"eval/std_episodic_return": wandb.plot.bar(table, "gravity", "std_episodic_return",
                                                              title="Std Episodic Return")})

        data = [[x, y] for (x, y) in zip(gravity_keys, min_gravity_returns)]
        table = wandb.Table(data=data, columns=["gravity", "min_episodic_return"])
        wandb.log({"eval/min_episodic_return": wandb.plot.bar(table, "gravity", "min_episodic_return",
                                                              title="Min Episodic Return")})

        data = [[x, y] for (x, y) in zip(gravity_keys, max_gravity_returns)]
        table = wandb.Table(data=data, columns=["gravity", "max_episodic_return"])
        wandb.log({"eval/max_episodic_return": wandb.plot.bar(table, "gravity", "max_episodic_return",
                                                              title="Max Episodic Return")})

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
