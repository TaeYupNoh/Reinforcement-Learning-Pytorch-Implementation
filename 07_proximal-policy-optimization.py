# proximal : 가까운
# policy를 업데이트 할 때, 기존의 policy와의 거리를 최소화하는 방향으로 업데이트

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import deque


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist


def ppo(env_name, num_steps, mini_batch_size, ppo_epochs, threshold_reward):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    policy = Policy(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    memory = deque()
    rewards = []
    steps = []
    total_steps = 0

    for i in range(num_steps):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            dist = policy(torch.FloatTensor(state))
            action = dist.sample().numpy()
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, 1 - done))
            state = next_state
            total_reward += reward
            step += 1
            total_steps += 1

            if total_steps % mini_batch_size == 0:
                update(memory, policy, optimizer, ppo_epochs, mini_batch_size)

        rewards.append(total_reward)
        steps.append(step)

        mean_reward = sum(rewards[-100:]) / 100
        if mean_reward > threshold_reward:
            print(f"Solved after {i} episodes")
            return rewards, steps

        print(
            f"Episode {i}: total steps = {total_steps}, reward = {total_reward}")

    return rewards, steps


def update(memory, policy, optimizer, ppo_epochs, mini_batch_size, clip_ratio=0.2, gamma=0.99, lmbda=0.95):
    rewards = []
    states = []
    actions = []
    next_states = []
    dones = []
    for state, action, reward, next_state, done in memory:
        states.append(torch.FloatTensor(state))
        actions.append(torch.FloatTensor(action))
        rewards.append(torch.FloatTensor([reward]))
        next_states.append(torch.FloatTensor(next_state))
        dones.append(torch.FloatTensor([done]))

    old_dist = policy(torch.stack(states)).detach()
    old_log_prob = old_dist.log_prob(torch.stack(actions)).detach()

    returns = []
    discounted_reward = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        if done:
            discounted_reward = 0
        discounted_reward = reward + gamma * discounted_reward
        returns.insert(0, discounted_reward)

    returns = torch.stack(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    for _ in range(ppo_epochs):
        for index in BatchSampler(RandomSampler(range(len(memory))), mini_batch_size, drop_last=False):
            batch_states = torch.stack([states[i] for i in index])
            batch_actions = torch.stack([actions[i] for i in index])
            batch_old_log_prob = torch.stack(
                [old_log_prob[i] for i in index]).detach()
            batch_returns = torch.stack([returns[i] for i in index])
            batch_advantage = torch.zeros(len(index), 1)
            for i in range(len(index)):
                dist = policy(batch_states[i])
                log_prob = dist.log_prob(batch_actions[i])
                ratio = torch.exp(log_prob - batch_old_log_prob[i])
                surr1 = ratio * batch_advantage[i]
                surr2 = torch.clamp(ratio, 1 - clip_ratio,
                                    1 + clip_ratio) * batch_advantage[i]
                batch_advantage[i] = torch.max(surr1, surr2)

            optimizer.zero_grad()
            batch_loss = -torch.min(batch_advantage, torch.clamp(ratio,
                                    1 - clip_ratio, 1 + clip_ratio) * batch_advantage).mean()
            batch_loss.backward()
            optimizer.step()
