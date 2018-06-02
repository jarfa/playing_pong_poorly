"""
Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
Made from cannibalizing https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
and https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

import argparse
import logging
from itertools import count

import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y.%m.%d %I:%M:%S",
    level=logging.DEBUG
)

class Policy(nn.Module):
    def __init__(self, D, H):
        super(Policy, self).__init__()
        self.hidden = torch.nn.Linear(D, H)
        self.out = torch.nn.Linear(H, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        h = F.relu(self.hidden(x))
        logp = self.out(h)
        return  F.softmax(logp, dim=1)#F.sigmoid(logp)

# DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda') # Uncomment this to run on GPU
ENV = gym.make('Pong-v0')
POLICY = Policy(H=200, D=80 * 80)
OPTIMIZER = optim.Adam(POLICY.parameters(), lr=1e-2)
EPSILON = np.finfo(np.float32).eps.item()

def prepro(I):
    """ Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
    This function is copied almost verbatim from Karpathy's code.
    """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def select_action(state):
    # TODO: should this be a method of POLICY?
    processed_state = prepro(state)
    tensor_state = torch.from_numpy(processed_state).float().unsqueeze(0)

    probs = POLICY(tensor_state)
    m = Categorical(probs)
    action = m.sample()
    POLICY.saved_log_probs.append(m.log_prob(action))

    # action is in {0,1}, but for gym it needs to be in {2,3}
    action_for_gym = action.item() + 2
    return action_for_gym


def finish_episode(gamma):
    # TODO: should this be a method of POLICY?
    R = 0
    policy_loss = []
    rewards = []
    for r in POLICY.rewards[::-1]:
        R = gamma * R +  r
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + EPSILON)
    for log_prob, reward in zip(POLICY.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    OPTIMIZER.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    OPTIMIZER.step()
    del POLICY.rewards[:]
    del POLICY.saved_log_probs[:]


def main(args):
    ENV.seed(args.seed)
    torch.manual_seed(args.seed)

    running_reward = None
    for i_episode in count(1):
        state = ENV.reset()
        match_reward = 0
        for t in range(10 ** 4):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = ENV.step(action)
            if args.render:
                ENV.render()
            POLICY.rewards.append(reward)
            match_reward += reward
            if done:
                break

        if running_reward is None:
            running_reward = match_reward
        else:
            running_reward = running_reward * 0.99 + match_reward * 0.01
        finish_episode(args.gamma)
        if i_episode % args.log_interval == 0:
            logging.info(
                "Episode %d\tLast Reward: %d\tEWMA Reward: %.2f",
                i_episode, match_reward, running_reward
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch REINFORCE example")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="G",
                        help="discount factor (default: 0.99)")
    parser.add_argument("--seed", type=int, default=543, metavar="N",
                        help="random seed (default: 543)")
    parser.add_argument("--render", action="store_true",
                        help="render the environment")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="interval between training status logs (default: 10)")
    main(parser.parse_args())
