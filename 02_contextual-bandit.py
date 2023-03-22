import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class ContextualBanditEnvironment:
    def __init__(self):
        self.states = np.array([[0.2,0,-0.0,-5], 
                                 [0.1,-5,1,0.25], 
                                 [-5,5,5,5]])
        self.n_states = self.states.shape[0]
        self.n_arms = self.states.shape[1]

    def get_state(self):
        self.state = np.random.randint(0, self.n_states)
        return self.state

    def pull_arm(self, action):
        bandit = self.states[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1

class Policy(nn.Module):
    def __init__(self, n_states, n_arms):
        super(Policy, self).__init__()
        self.n_states = n_states
        self.n_arms = n_arms
        self.layer1 = nn.Linear(self.n_states, self.n_arms)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.softmax(x)
        return x
    
    def get_action(self, state):
        state_one_hot = np.zeros(n_states)
        state_one_hot[state] = 1
        state_one_hot = torch.Tensor(state_one_hot)
        probs = self.forward(state_one_hot)
        # probs의 예 :[0.25, 0.25, 0.25, 0.25]
        return probs
    
    def update(self, action_probs, action, reward, optimizer):
        log_prob = torch.log(action_probs)[action]
        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Define hyperparameters
n_episodes = 10000
learning_rate = 0.1
        
# Initialize bandit and policy
env = ContextualBanditEnvironment()
policy = Policy(env.n_states, env.n_arms)
optimizer = optim.SGD(policy.parameters(), lr=learning_rate)

# states, states 수, states별 arms 수 설정
states = env.states
n_states = env.n_states
n_arms = env.n_arms
# 밴딧에 대한 점수판 0으로 설정
total_reward = np.zeros_like(states)
# 랜덤한 액션을 취할 가능성 설정
e = 0.1

# Training loop
for episode in range(n_episodes):
    state = env.get_state()
    # action_probs는 4개의 arms에 대한 softmax값 
    # 예) ([0.25, 0.25, 0.25, 0.25])
    action_probs = policy.get_action(state)
    
    # action을 선택해야 하는데 두 가지 경우의 수가 있음
    # 랜덤한 액션을 선택하거나, Feed-forward를 통해 얻은 action을 선택
    if np.random.rand(1) < e:
        action = np.random.randint(0, n_states)
    else:     

        action = torch.multinomial(action_probs, num_samples=1) # 0-3 중 하나
    
    reward = env.pull_arm(action) # 1 or -1
    total_reward[state, action] += reward
    
    # Calculate loss and update policy
    policy.update(action_probs, action, reward, optimizer)
        
    # Print total reward of each episode
    if episode % 500 == 0:
        print("Mean reward for each of the 3 bandits : {}".format(np.mean(total_reward, axis = 1)))

print("Testing policy...")
for state in range(n_states):
    action_probs = policy.get_action(state)
    action = torch.multinomial(action_probs, num_samples=1)
    print(f"The agent(Policy) thinks action {action+1} for bandit {state+1} is the most promising....")

    if np.argmax(action_probs.detach().numpy()) == np.argmin(states[state]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")