import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ContextualBandit:
    def __init__(self):
        self.state = np.array([[1.0, 2.0, 3.0, 4.0],
                               [4.0, 3.0, 2.0, 1.0],
                               [1.0, 2.0, 3.0, 4.0]])
        self.n_bandits, self.n_features = self.state.shape
        self.n_actions = 3

        self.correct_bandit = np.random.randint(self.n_bandits)

    def get_reward(self, action):
        reward = 0
        if action == self.correct_bandit:
            reward = 1.0
        return reward


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.Tensor(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=0)


def choose_action(policy, state):
    state = torch.Tensor(state)
    action_probs = policy.forward(state)
    action = np.random.choice(range(policy.n_actions),
                              p=action_probs.detach().numpy())
    return action


def update_policy(policy, optimizer, rewards, actions, states):
    loss = torch.Tensor(
        [-np.log(action_probs[a]) * r for a, r in zip(actions, rewards)])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    n_episodes = 2000
    learning_rate = 0.01
    bandit = ContextualBandit()
    policy = Policy()
    optimizer = optim.SGD(policy.parameters(), lr=learning_rate)

    for episode in range(n_episodes):
        state = bandit.state[bandit.correct_bandit]
        action = choose_action(policy, state)
        reward = bandit.get_reward(action)

        update_policy(policy, optimizer, [reward], [action], [state])

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}")


if __name__ == '__main__':
    main()
