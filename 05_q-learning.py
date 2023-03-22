import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        return x

# Create the Q-Network and the optimizer
q_network = QNetwork()
optimizer = optim.Adam(q_network.parameters(), lr=0.01)

# Set learning parameters
# gamma는 discount rate를 나타냄
gamma = 0.99
e = 0.1
num_episodes = 5000
jList = []
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    s, _ = env.reset()
    rAll = 0
    done = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # Choose an action by greedily (with e chance of random action) from the Q-network
        # np.identity는 대각선만 1로 채운 행렬을 생성 -> 16개의 행에 대해 one-hot encoding 해주는 역할
        Q = q_network(torch.from_numpy(np.identity(16)[s:s+1]).float())
        # exploration 하는 경우엔 a를 랜덤 값으로 (현재 e가 0.1이므로 10%의 확률로 랜덤한 행동을 취함)
        if np.random.rand(1) > e:
            a = torch.argmax(Q).item()
        else:
            a = env.action_space.sample()
        # Get new state and reward from environment
        s1, r, done, _, _ = env.step(a)
        # Obtain the Q' values by feeding the new state through our network
        Q1 = q_network(torch.from_numpy(np.identity(16)[s1:s1+1]).float())
        # Obtain maxQ' and set our target value for chosen action.
        maxQ1 = torch.max(Q1).item()
        targetQ = Q.clone()
        # 벨만 방정식
        targetQ[0, a] = r + gamma*maxQ1
        # Train our network using target and predicted Q values
        loss = nn.MSELoss()(Q, targetQ)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rAll += r
        s = s1
        if done == True:
            # Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

print("Percent of successful episodes: " + str(sum(rList)/num_episodes*100) + "%")

# plt.plot(rList)
# plt.show()

# plt.plot(jList)
# plt.show()