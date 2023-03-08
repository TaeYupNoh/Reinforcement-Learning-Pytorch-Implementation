# contextual bandit -> 여러 개의 bandits를 가정해, 상태(state)라는 개념 도입

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define contextual bandit
class ContextualBandit:
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[0.2,0,-0.0,-5], 
                                 [0.1,-5,1,0.25], 
                                 [-5,5,5,5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        
    def get_state(self):
        self.state = np.random.randint(0, self.num_bandits)
        return self.state
    
    def pull_arm(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1

# Define policy
class Policy(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):
        super(Policy, self).__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(num_features, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_actions)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x
    
    def get_action(self, state):
        state = torch.Tensor(state).unsqueeze(0)
        probs = self.forward(state)
        action = torch.multinomial(probs, num_samples=1)
        return action.item() # 0-3 중 하나

# Define hyperparameters
n_episodes = 5000
n_steps_per_episode = 1000
learning_rate = 0.1
num_features = 4
num_actions = 3
hidden_size = 10

# Initialize bandit and policy
bandit = ContextualBandit()
policy = Policy(num_features, num_actions, hidden_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Training loop
for episode in range(n_episodes):
    state = bandit.get_state()
    total_reward = 0
    for step in range(n_steps_per_episode):
        action = policy.get_action(state) # 0-3 중 하나의 값
        reward = bandit.pull_arm(action)
        total_reward += reward
        
        # Calculate loss and update policy
        optimizer.zero_grad()
        
        state_one_hot = np.zeros(n_states)
        state_one_hot[state] = 1
        state_one_hot = torch.Tensor(state_one_hot)
        
        action_one_hot = np.zeros(n_arms)
        action_one_hot[action] = 1
        action_one_hot = torch.Tensor(action_one_hot).unsqueeze(0)
        
        action_probs = policy.forward(state_one_hot)
        action_prob = action_probs.gather(1, action_one_hot.long())
        
        loss = -torch.log(action_prob) * reward
        loss.backward()
        optimizer.step()
        
        state = bandit.get_state()
        
    # Print total reward of each episode
    if episode % 100 == 0:
        print("Episode {}: Total reward = {}".format(episode, total_reward))

print("Testing policy...")
for state in range(num_features):
    action = policy.get_action(state)
    print("State {}: Action = {}".format(state, action))