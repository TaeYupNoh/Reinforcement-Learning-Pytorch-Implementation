import numpy as np
import torch
import torch.optim as optim

class BanditEnvironment:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.means = np.random.normal(0, 1, size=n_arms)
        self.stds = np.ones(n_arms)
    
    def pull(self, arm):
        return np.random.normal(self.means[arm], self.stds[arm])

class SoftmaxPolicy:
    def __init__(self, n_arms=10, temperature=0.1):
        self.n_arms = n_arms
        self.temperature = temperature
        self.weights = torch.zeros(n_arms, requires_grad=True)
    
    def __call__(self):
        probs = torch.softmax(self.weights / self.temperature, dim=0)
        return probs
    
    def update(self, arm, reward, optimizer):
        log_prob = torch.log_softmax(self.weights / self.temperature, dim=0)[arm]
        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Hyperparameters
n_episodes = 1000
n_steps_per_episode = 1000
learning_rate = 0.01

# Initialize environment and policy
env = BanditEnvironment()
policy = SoftmaxPolicy()

# Define optimizer
optimizer = optim.SGD([policy.weights], lr=learning_rate)

# Training loop
for episode in range(n_episodes):
    total_reward = 0
    for step in range(n_steps_per_episode):
        # Choose arm according to policy
        probs = policy()
        arm = torch.multinomial(probs, num_samples=1).item()
        
        # Receive reward from environment
        reward = env.pull(arm)
        total_reward += reward
        
        # Update policy
        policy.update(arm, reward, optimizer)
    
    # Print total reward for episode
    print(f"Episode {episode}: Total reward = {total_reward}")
