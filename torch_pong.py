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

ENV = gym.make('Pong-v0')
EPSILON = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    def __init__(self, D, H):
        super(Policy, self).__init__()
        self.hidden = torch.nn.Linear(D, H)
        self.out = torch.nn.Linear(H, 2)

        self.saved_log_probs = []
        self.rewards = []
        self.num_frames = 0

    def forward(self, x):
        self.num_frames += 1

        h = F.relu(self.hidden(x))
        logp = self.out(h)
        return F.softmax(logp, dim=1)

    def select_action(self, state):
        processed_state = prepro(state)
        tensor_state = torch.from_numpy(processed_state).float().unsqueeze(0).cuda()

        probs = self.__call__(tensor_state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))

        # action is in {0,1}, but for gym it needs to be in {2,3}
        return action.item() + 2

    def finish_episode(self, objective, gamma, optimizer):
        R = 0
        policy_loss = []
        if objective in ("win", "lose"):
            rewards = []
            for r in self.rewards[::-1]:
                R = gamma * R +  r
                rewards.insert(0, R)

            rewards = torch.tensor(rewards).cuda()
            rewards = (rewards - rewards.mean()) / (rewards.std() + EPSILON)

            mult = -1 if objective == "win" else 1
            for log_prob, reward in zip(self.saved_log_probs, rewards):
                policy_loss.append(mult * log_prob * reward)

            final_loss = torch.cat(policy_loss).sum()
        elif objective == "length":
            # Our loss function is -num_frames
            final_loss = -1. * self.num_frames * sum(self.saved_log_probs)
            #TODO: will this work?
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        self.num_frames = 0
        del self.rewards[:]
        del self.saved_log_probs[:]


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


def update_mean(n, old_mean, new_data):
    return ((n - 1) * old_mean + new_data) / n


def main(args):
    ENV.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = Policy(H=200, D=80 * 80).cuda()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    ewma_reward = 0
    ewma_frames = 0
    for i_episode in count(1):
        state = ENV.reset()
        match_reward = 0
        for _ in range(10 ** 4):  # Don't infinite loop while learning
            action = policy.select_action(state)
            state, reward, done, _ = ENV.step(action)
            if args.render:
                ENV.render()
            policy.rewards.append(reward)
            match_reward += reward
            if done:
                break

        # Take the first 20 episodes to 'seed' the EWMA, then do it the normal way
        seed_episodes = 20
        if i_episode <= seed_episodes:
            ewma_reward = update_mean(i_episode, ewma_reward, match_reward)
            ewma_frames = update_mean(i_episode, ewma_frames, policy.num_frames)
        else:
            ewma_reward = ewma_reward * 0.99 + match_reward * 0.01
            ewma_frames = ewma_frames * 0.99 + policy.num_frames * 0.01

        if i_episode % args.log_interval == 0:
            logging.info(
                "Episode %d\tLast Reward: %d\tEWMA Reward: %.2f\t"
                "Frames: %d\tEWMA Frames: %.2f",
                i_episode, match_reward, ewma_reward, policy.num_frames,
                ewma_frames
            )
        policy.finish_episode(args.objective, args.gamma, optimizer)

        # only save when i_episode is a power of 2, but skip the range [2,16]
        if (args.save_path and (i_episode & (i_episode - 1) == 0) and
            (i_episode < 2 or i_episode > 16)):
            info = {
                "episode": i_episode,
                "arch": args.__dict__,
                "state_dict": policy.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "ewma_frames": ewma_frames,
                "ewma_reward": ewma_reward,
            }
            filename = "{base}_{i:06d}.pth.tar".format(
                base=args.save_path, i=i_episode)
            with open(filename, "wb") as f:
                torch.save(info, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Would you like to play a game?")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="L",
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="G",
                        help="discount factor (default: 0.99)")
    parser.add_argument("--seed", type=int, default=543, metavar="N",
                        help="random seed (default: 543)")
    parser.add_argument("--render", action="store_true",
                        help="render the environment")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="interval between training status logs (default: 10)")
    parser.add_argument("--objective", type=str, default="win",
                        choices=["win", "lose", "length"],
                        help="What's the objective of our RL agent?")
    parser.add_argument("--save_path", type=str, default=None, metavar="F",
                        help="Base path to save model parameters to (optional).")

    main(parser.parse_args())
