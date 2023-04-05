# vanilla policy gradient는 앞의 두 bandit 문제에서 고려한 것을 포함해 총 3가지를 고려해야 함
# 1. 액션 의존성 :MAB에서 사용된 것처럼 각각의 액션이 보상을 가져다 줄 확률은 다름
# 2. 상태 의존성 :MAB와 달리, CB에서 각 액션을 취할 때의 보상은 그 액션을 취할 당시의 상태와 관계가 있음
# 3. 시간 의존성 :에이전트는 보상에 대해 시간 지연된 시점에 학습함, 따라서 데이터에 샘플을 저장 후 뭉쳐서 학습

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.002
gamma = 0.99

# 여기서 Policy는 상태가치함수를 나타냄


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


env = gym.make('CartPole-v1')
pi = Policy()
score = 0.0
print_interval = 100

for n_epi in range(1000):
    # s는 state로, Cart Position / Cart Velocity / Pole Angle / Pole Velocity 정보를 담고 있음
    s, _ = env.reset()
    done = False

    while not done:
        # policy는 state(4)를 받아 action(2) 별 확률을 반환
        prob = pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample()
        s_prime, r, done, info, _ = env.step(a.item())
        pi.put_data((r, prob[a]))
        s = s_prime
        score += r

    pi.train_net()

    if n_epi % print_interval == 0 and n_epi != 0:
        print("# of episode :{}, avg score : {}".format(
            n_epi, score/print_interval))

        # 성공 조건 설정
        if score/print_interval > 500:
            print("Congratulations! solved :)")
            break

        score = 0.0

env.close()
