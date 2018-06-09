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


class Policy(nn.Module):
    def __init__(self, D, H):
        super(Policy, self).__init__()
        self.hidden = torch.nn.Linear(D, H)
        self.out = torch.nn.Linear(H, 2)

        self.saved_log_probs = []
        self.rewards = []
        self.num_frames = []

    def forward(self, x):
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

        # action is in {0,1}, but for the gym API it needs to be in {2,3}
        return action.item() + 2

    def finish_episode(self, objective, gamma, optimizer):
        # assert self.num_frames[-1] == 0  # remove after the sanity test
        # del self.num_frames[-1]  #this should be a trailing 0
        game_rewards = np.array([r for r in self.rewards if r != 0])

        win_rate = sum(game_rewards == 1) / len(game_rewards)
        frames_per_game = sum(self.num_frames) / len(self.num_frames)

        if objective in ("win", "lose"):
            mult = -1.0 if objective == "win" else 1.0
            # The rewards vector is mostly 0s, with -1s and 1s marking where
            # individual points were scored. We're iterating through it
            # backwards, decaying rewards up to where points were scored.
            R = 0
            rewards_to_learn = []
            for r in self.rewards[::-1]:
                # If a point was scored, don't pull the decayed reward from the
                # next frame - just use the score
                R = (gamma * R +  r) if r == 0 else r
                rewards_to_learn.insert(0, mult * R)

        elif objective == "length":
            raise NotImplementedError("Implement me!!!")

        rewards_to_learn = torch.tensor(rewards_to_learn).cuda()
        rewards_to_learn = (rewards_to_learn - rewards_to_learn.mean()
                            ) / (rewards_to_learn.std() + EPSILON)

        policy_loss = torch.dot(
            torch.cat(self.saved_log_probs).cuda(),
            rewards_to_learn
        )
        final_loss = policy_loss.sum()

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        del self.rewards[:]
        del self.num_frames[:]
        del self.saved_log_probs[:]

        return win_rate, frames_per_game


def update_mean(n, old_mean, new_data):
    return ((n - 1) * old_mean + new_data) / n


def main(args):
    ENV.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = Policy(H=200, D=80 * 80).cuda()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    ewma_win_rate = 0
    ewma_frames = 0
    current_frames = 0
    for i_episode in count(1):
        state = ENV.reset()
        for _ in range(10 ** 4):  # Don't infinite loop while learning
            action = policy.select_action(state)
            state, reward, done, _ = ENV.step(action)
            if args.render:
                ENV.render()
            policy.rewards.append(reward)
            current_frames += 1
            if reward != 0:
                policy.num_frames.append(current_frames)
                current_frames = 0
            if done:
                break

        win_rate, frames_per_game = policy.finish_episode(
            args.objective, args.gamma, optimizer)
        # Take the first 20 episodes to 'seed' the EWMA, then do it the normal way
        if i_episode <= 20:
            ewma_win_rate = update_mean(i_episode, ewma_win_rate, win_rate)
            ewma_frames = update_mean(i_episode, ewma_frames, frames_per_game)
        else:
            ewma_win_rate = ewma_win_rate * 0.99 + win_rate * 0.01
            ewma_frames = ewma_frames * 0.99 + frames_per_game * 0.01

        if i_episode % args.log_interval == 0:
            logging.info(
                "Episode %d\tLast Win Rate: %.2f\tEWMA Win Rate: %.2f\t"
                "Frames/Game: %d\tEWMA Frames/Game: %.1f",
                i_episode, win_rate, ewma_win_rate, frames_per_game,
                ewma_frames
            )


        # only save when i_episode is a power of 2, but skip the range [2,16]
        if (args.save_path and (i_episode & (i_episode - 1) == 0) and
            (i_episode < 2 or i_episode > 16)):
            info = {
                "episode": i_episode,
                "arch": args.__dict__,
                "state_dict": policy.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "ewma_frames": ewma_frames,
                "ewma_win_rate": ewma_win_rate,
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
