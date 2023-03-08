import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

env = gym.make('CartPole-v0')

gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent(nn.Module):
    def __init__(self, lr, s_size, a_size, h_size):
        super(agent, self).__init__()
        self.lr = lr
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = nn.Linear(s_size, h_size)
        self.hidden = nn.Linear(h_size, h_size)
        self.output = nn.Linear(h_size, a_size)
        self.softmax = nn.Softmax(dim=1)
        
        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = []
        self.action_holder = []

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.state_in(x)
        x = torch.relu(x)
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

    def update(self, rewards, actions, states):
        # Compute discounted rewards
        discounted_rewards = discount_rewards(rewards)

        # Convert actions to tensor
        actions = torch.tensor(actions, dtype=torch.long)

        # Convert rewards to tensor
        rewards = torch.tensor(discounted_rewards, dtype=torch.float)

        # Convert states to tensor
        states = torch.tensor(states, dtype=torch.float)

        # Compute the negative log-likelihood loss
        logits = self(states)
        loss = self.loss(logits, actions) * -rewards.mean()

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

torch.manual_seed(0) # Set the random seed for reproducibility

myAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8) # Load the agent.

total_episodes = 5000 # Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

total_reward = []
total_length = []

for i in range(total_episodes):
    s = env.reset()
    running_reward = 0
    ep_history = {'states': [], 'actions': [], 'rewards': []}
    for j in range(max_ep):
        # Probabilistically pick an action given our network outputs.
        a_dist = myAgent(torch.tensor(s, dtype=torch.float))
        a = np.random.choice(a_dist.detach().numpy(), p=a_dist.detach().numpy())
        a = np.argmax(a_dist.detach().numpy() == a)
        s1, r, d, _ = env.step(a) # Get our reward for taking an action given a bandit.
        ep_history['states'].append(s)
        ep_history['actions'].append(a)
        ep_history['rewards'].append(r)
        s = s1
        running_reward += r

        if d == True:
            # Update the network
            myAgent.update(ep_history['rewards'], ep_history['actions'], ep_history['states'])

            # Reset the episode history
            ep_history = {'states': [], 'actions': [], 'rewards': []}

            # Record the total reward and episode length
            total_reward.append(running_reward)
            total_length.append(j)
            break

        if j == max_ep-1:
            # Update the network
            myAgent.update(ep_history['rewards'], ep_history['actions'], ep_history['states'])

            # Reset the episode history
            ep_history = {'states': [], 'actions': [], 'rewards': []}

            # Record the total reward and episode length
            total_reward.append(running_reward)
            total_length.append(j)
            break

        if i % update_frequency == 0:
            # Get the gradients from the agent
            grads = myAgent.parameters()

            # Add the gradients to the gradBuffer
            for ix, grad in enumerate(grads):
                gradBuffer[ix] += grad.detach().numpy()

            # Update the agent's parameters with the mean of the gradients
            for ix, grad in enumerate(gradBuffer):
                myAgent.parameters()[ix].grad = torch.tensor(grad / update_frequency, requires_grad=True)

            # Update the network
            myAgent.optimizer.step()

            # Reset the gradBuffer
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

    # Print the total reward every 100 episodes
    if i % 100 == 0:
        print("Episode: ", i, " Total Reward: ", np.mean(total_reward[-100:]))

print("Training complete!")