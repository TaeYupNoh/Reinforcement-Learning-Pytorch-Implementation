# multi-armed bandit -> 하나의 bandit인데 여러 개의 arms

import numpy as np
import torch
import torch.optim as optim

class BanditEnvironment:
    # 밴딧 개수 4개로 설정
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # self.bandits는 밴딧마다 임의의 값을 지정해 줌. 이 값을 이용해 어느 밴딧을 당겨야 가장 큰 보상이 올 지 정할 예정
        # 여기서 self.arms[3], 즉 네 번째 밴딧의 값이 가장 작다(-1) 
        # 즉, 랜덤한 값 임의추출 했을 때 그 값보다 밴딧의 값([0.2,0,-0.2,-2])이 작을 확률이 가장 높다.
        # 네 번째 밴딧을 자주 당기면 보상이 최대화되는 것
        self.arms = np.array([0.2,0,-0.2,-1])
    
    def pull(self, arm):
        # 0-1 사이의 랜덤한 값을 구한다 
        result = np.random.randn(1)
        if result > self.arms[arm]:
            return 1
        else:
            return -1

class SoftmaxPolicy:
    def __init__(self, n_arms, temperature):
        self.n_arms = n_arms
        # temperature 값이 높을수록 무작위성이 커지고 안정되지 않은 선택을 할 확률이 강조됨
        # temperature 값이 0에 가까워질수록 정책에 따른 선택이 거의 절대적으로 이루어짐 
        # 즉, 탐험(exploration)을 하지 않고 안정적인 선택(exploitation)을 더 많이 함
        # 반대로 temperature 값이 커지면 탐험을 더 많이 함(log_softmax에서 각 weights의 확률 값 차이가 적어짐) 
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

# 밴딧 수, temperature 설정
n_arms = 4
temperature = 10

# Hyperparameters
n_episodes = 50
n_steps_per_episode = 1000
learning_rate = 0.01

# Initialize environment and policy
env = BanditEnvironment(n_arms)
policy = SoftmaxPolicy(n_arms, temperature)

# Define optimizer
optimizer = optim.SGD([policy.weights], lr=learning_rate)

# Training loop
for episode in range(n_episodes):
    total_reward = np.zeros(n_arms)
    for step in range(n_steps_per_episode):
        # Choose arm according to policy
        probs = policy()
        # 다항분포에서 num_samples 개수의 표본을 생성
        # .items()를 통해 Tensor 값을 int로 변환
        # 액션 할 arm 번호 선택 (0-n_arms 중 하나의 값)
        arm = torch.multinomial(probs, num_samples=1).item()

        # Receive reward from environment
        reward = env.pull(arm)
        total_reward[arm] += reward
        
        # Update policy
        policy.update(arm, reward, optimizer)
    
    # Print total reward for episode
    print(f"Episode {episode+1}: Total reward = {total_reward}")

print("Testing policy...")

print(f"The agent(Policy) thinks action {np.argmax(total_reward) + 1} is the most promising....")

if np.argmax(total_reward) == np.argmin(env.arms):
    print("...and it was right!")
else:
    print("...and it was wrong!")