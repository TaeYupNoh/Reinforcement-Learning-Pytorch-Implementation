import gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

Q = np.zeros([env.observation_space.n, env.action_space.n])

# hyperparameters
lr = .85
# discount factor (할인 인자가 크면 미래의 보상을 더 중요시하게 여겨 학습이 안정적이나, 느려짐)
y = .99
num_episodes = 5000

# 보상의 총계를 담을 리스트 생성
rList = []
for i in range(num_episodes):
    # 환경과 상태 초기화
    s, _ = env.reset()
    rAll = 0
    d = False
    j = 0
    # Q-Table을 이용한 행동 선택
    while j < 99:
        j+=1
        # 현재 상태에 대한 행동 선택
        # np.random.randn(1,env.action_space.n) : *행동의 개수*만큼의 랜덤한 값을 생성
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # 선택한 행동으로 환경에서 한 타임스텝 진행
        s1, r, d, _, _ = env.step(a)
        # Q-Table 업데이트 (by 벨만 방정식 : r + y*max(Q[s1,:]))
        # y*np.max(Q[s1,:]) - Q[s,a] : TD error(Temporal Difference error)
        # TD error가 크면 현재의 행동이 미래에 더 큰 보상을 가져올 것이라고 예상하고, 그만큼 큰 변화를 가질 수 있게 함
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    rList.append(rAll)
    
print("Success rate: " + str(sum(rList)/num_episodes))

print("Final Q-Table Values")
print(Q)