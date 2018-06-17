"""
Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
Made from cannibalizing https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
and https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

import argparse
import logging
from itertools import count
from sys import exit

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

EPSILON = np.finfo(np.float32).eps.item()


def update_mean(n, old, new):
    return ((n - 1) * old + new) / n


def update_ewma(old, new, alpha=0.05):
    return (1. - alpha) * old + alpha * new


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
    def __init__(self, D, H, gpu):
        super(Policy, self).__init__()
        self.hidden = torch.nn.Linear(D, H)
        self.out = torch.nn.Linear(H, 2)
        self.gpu = gpu

        self.saved_log_probs = []
        self.rewards = []
        self.num_frames = []

    def forward(self, x):
        h = F.relu(self.hidden(x))
        logp = self.out(h)
        return F.softmax(logp, dim=1)

    def select_action(self, state):
        processed_state = prepro(state)
        tensor_state = torch.from_numpy(processed_state).float().unsqueeze(0)
        if self.gpu:
            tensor_state = tensor_state.cuda()

        probs = self.__call__(tensor_state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))

        # action is in {0,1}, but for the gym API it needs to be in {2,3}
        return action.item() + 2

    def finish_episode(self, objective, gamma, optimizer):
        game_rewards = np.array([r for r in self.rewards if r != 0])

        win_rate = sum(game_rewards == 1) / len(game_rewards)
        frames_per_game = sum(self.num_frames) / len(self.num_frames)

        N = len(self.rewards)
        rewards_to_learn = [0] * N
        if objective in ("win", "lose"):
            mult = -1.0 if objective == "win" else 1.0
            # The rewards vector is mostly 0s, with -1s and 1s marking where
            # individual points were scored. We're iterating through it
            # backwards, decaying rewards up to where points were scored.
            R = 0
            for i in reversed(range(N)):
                r = self.rewards[i]
                R = gamma * R if r == 0 else r
                rewards_to_learn[i] = mult * R

        elif objective == "length":
            match_index = len(self.num_frames) - 1 #start by looking at the last game
            for i in reversed(range(N)):
                if i < (N - 1) and self.rewards[i] == 0:
                    # This frame didn't have a point scored, so pull the
                    # eventual number of frames from the next frame
                    rewards_to_learn[i] = rewards_to_learn[i + 1]
                    # I don't think we should be decaying the reward for this
                    # type of learning, all moves made were equally important
                    # in prolonging the match.
                else:
                    # rewards != 0 marks a point scored, so record the number of
                    # frames it took
                    rewards_to_learn[i] = -1.0 * self.num_frames[match_index]
                    match_index -= 1

        rewards_to_learn = torch.tensor(rewards_to_learn)
        if self.gpu:
            rewards_to_learn = rewards_to_learn.cuda()
        rewards_to_learn = (rewards_to_learn - rewards_to_learn.mean()
                            ) / (rewards_to_learn.std() + EPSILON)

        policy_loss = torch.dot(
            torch.cat(self.saved_log_probs),
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


def main(args):
    environment = gym.make('Pong-v0')
    environment.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = Policy(H=200, D=80 * 80, gpu=args.gpu)
    if args.gpu:
        policy = policy.cuda()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    ewma_win_rate = 0
    ewma_frames = 0
    current_frames = 0
    for i_batch in count(1):
        state = environment.reset()
        for _ in range(args.minibatch):
            # args.minibatch is the number of games to play. Let's define a game
            # as ending when a point is scored by either team. The default gym
            # behavior is to play until a team has 21 points, but we're not
            # using that.
            for i in range(10 ** 4):  # to avoid infinite loops
                action = policy.select_action(state)
                state, reward, done, _ = environment.step(action)
                if args.render:
                    environment.render()
                policy.rewards.append(reward)
                current_frames += 1
                if done:
                    state = environment.reset()
                if reward != 0:
                    break

            policy.num_frames.append(current_frames)
            current_frames = 0

        win_rate, frames_per_game = policy.finish_episode(
            args.objective, args.gamma, optimizer)
        # Take the first 20 episodes to 'seed' the EWMA, then do it the normal way
        if i_batch <= 20:
            ewma_win_rate = update_mean(i_batch, ewma_win_rate, win_rate)
            ewma_frames = update_mean(i_batch, ewma_frames, frames_per_game)
        else:
            ewma_win_rate = update_ewma(ewma_win_rate, win_rate)
            ewma_frames = update_ewma(ewma_frames, frames_per_game)

        last_episode = args.num_batches and i_batch == args.num_batches
        if last_episode or i_batch % args.log_interval == 0:
            logging.info(
                "Batch #%d\tLast Win Rate: %.2f\tEWMA Win Rate: %.2f\t"
                "Frames/Game: %d\tEWMA Frames/Game: %.1f",
                i_batch, win_rate, ewma_win_rate,
                frames_per_game, ewma_frames
            )

        # only save when at the end, or when i_batch is a power of 2 (but skip
        # the range [2,16])
        power_of_two = i_batch & (i_batch - 1) == 0
        if (
            args.save_path and (last_episode or
            (power_of_two and (i_batch < 2 or i_batch > 16)))
        ):
            info = {
                "batch": i_batch,
                "arch": args.__dict__,
                "state_dict": policy.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "ewma_frames": ewma_frames,
                "ewma_win_rate": ewma_win_rate,
            }
            filename = "{base}_{i:06d}.pth.tar".format(
                base=args.save_path, i=i_batch)
            with open(filename, "wb") as f:
                torch.save(info, f)

        if last_episode:
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shall we play a game?")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="L",
                        help="learning rate (default: 1e-3)")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, metavar="G",
                        help="discount factor (default: 0.99)")
    parser.add_argument("-s", "--seed", type=int, default=543, metavar="N",
                        help="random seed (default: 543)")
    parser.add_argument("-m", "--minibatch", type=int, default=25, metavar="M",
                        help="Minibatch size (default: 25)")
    parser.add_argument("-n", "--num_batches", type=int, metavar="N",
                        default=None, help="Maximum # of batches (default: inf)")
    parser.add_argument("--render", action="store_true",
                        help="render the environment")
    parser.add_argument("--cpu", dest="gpu", action="store_false",
                        help="run on the CPU, don't use cuda (default: False)")
    parser.add_argument("-i", "--log-interval", type=int, default=10, metavar="N",
                        help="interval between training status logs (default: 10)")
    parser.add_argument("-o", "--objective", type=str, default="win",
                        choices=["win", "lose", "length"],
                        help="What's the objective of our RL agent?")
    parser.add_argument("--save-path", type=str, default=None, metavar="F",
                        help="Base path to save model parameters to (optional).")

    main(parser.parse_args())
