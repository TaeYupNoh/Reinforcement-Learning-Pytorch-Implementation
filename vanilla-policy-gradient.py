import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import gym

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.softmax(self.output(x), dim=-1)
        return x

class Agent:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, batch_size):
        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.memory = []
        self.batch_size = batch_size
        
    def get_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy_network(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.memory.append((state, action, None, None))
        return action.item()
    
    def update_policy(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones = zip(*batch)
        
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)
        
        log_probs = torch.log(self.policy_network(states))
        log_probs_for_actions = torch.gather(log_probs, dim=1, index=actions.unsqueeze(1)).squeeze()
        future_rewards = torch.zeros_like(rewards)
        
        for i in range(len(rewards)-2, -1, -1):
            future_rewards[i] = rewards[i] + future_rewards[i+1] * (1 - dones[i])
        
        loss = - (log_probs_for_actions * future_rewards).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.memory = []


env = gym.make('CartPole-v1')
agent = Agent(input_dim=env.observation_space.shape[0], 
              hidden_dim=128, 
              output_dim=env.action_space.n, 
              lr=0.001,
              batch_size=16)  # 배치 사이즈 설정
n_episodes = 1000

for i in range(n_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0  # 에피소드에서 얻은 보상의 총합
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.memory[-1] = (state, action, reward, done)  # 마지막 transition에 보상과 done 추가
        
        state = next_state
        total_reward += reward
        
        if done:  # 에피소드가 끝나면 에이전트 학습
            agent.update_policy()
        
    if i % 100 == 0:
        print(f"Episode {i+1}: Total Reward = {total_reward}")
