

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:
    import gym



class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.fc(obs)
        mu = self.mu(x)
        value = self.v(x).squeeze(-1)
        std = torch.exp(self.log_std)
        return mu, std, value

    def act(self, obs):
        """
        Take a single observation (np.array [obs_dim]) and:
        - sample an action
        - return (action, log_prob, value)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  
        mu, std, value = self.forward(obs_t)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return (
            action.squeeze(0).detach().numpy(),  
            log_prob.item(),
            value.item(),
        )



class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
    ):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.net = ActorCritic(obs_dim, act_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def act(self, obs):
        """Thin wrapper so training loop can call agent.act(...)."""
        return self.net.act(obs)

    def compute_returns(self, rewards, dones, last_value):
        """
        Simple discounted returns (no GAE).
        rewards, dones: lists for a rollout
        last_value: value estimate for last state (bootstrap)
        """
        R = last_value
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0  
            R = r + self.gamma * R
            returns.append(R)
        returns.reverse()
        return np.array(returns, dtype=np.float32)

    def update(self, batch):
        obs = torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32)
        actions = torch.as_tensor(np.array(batch["actions"]), dtype=torch.float32)
        old_log_probs = torch.as_tensor(
            np.array(batch["log_probs"]), dtype=torch.float32
        )
        returns = torch.as_tensor(np.array(batch["returns"]), dtype=torch.float32)

        mu, std, values = self.net(obs)
        dist = torch.distributions.Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(-1)

        advantages = returns - values.detach()

        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.mean(torch.min(unclipped, clipped))

        value_loss = torch.mean((returns - values) ** 2)

        loss = policy_loss + 0.5 * value_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()



def make_env():
    """
    Replace this with your CARLA environment later.

    For now we use CarRacing-v2 just to make sure PPO runs.
    """
    return gym.make("CarRacing-v2", render_mode=None)


def train(
    total_steps=50_000,
    rollout_steps=1024,
):
    env = make_env()

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    obs_dim = int(np.prod(obs_shape))
    act_dim = act_shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high

    agent = PPO(obs_dim, act_dim)

    obs, info = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    ep_return = 0.0
    ep = 0
    step = 0

    while step < total_steps:
        batch = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
        }

        for _ in range(rollout_steps):
            flat_obs = obs.astype(np.float32).ravel()
            action, log_prob, value = agent.act(flat_obs)

            action = np.clip(action, act_low, act_high)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            batch["obs"].append(flat_obs)
            batch["actions"].append(action)
            batch["log_probs"].append(log_prob)
            batch["rewards"].append(reward)
            batch["dones"].append(done)

            ep_return += reward
            step += 1
            obs = np.asarray(next_obs, dtype=np.float32)

            
            if done:
                ep += 1
                print(f"Episode {ep}: return = {ep_return:.2f}, steps so far = {step}")
                with open("ppo_rewards.txt", "a") as f:
                    f.write(f"{ep},{ep_return}\n")

                ep_return = 0.0
                obs, info = env.reset()
                obs = np.asarray(obs, dtype=np.float32)

            if step >= total_steps:
                break


        with torch.no_grad():
            flat_obs = obs.astype(np.float32).ravel()
            obs_t = torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
            _, _, last_value = agent.net(obs_t)
            last_value = last_value.item()

        returns = agent.compute_returns(batch["rewards"], batch["dones"], last_value)
        batch["returns"] = returns

        agent.update(batch)

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    train()
