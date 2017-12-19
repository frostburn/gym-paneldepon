import sys

import gym
import gym.envs.registration
from gym import spaces
import numpy as np  # noqa: I001
from six import StringIO

from gym_paneldepon.bitboard import HEIGHT, WIDTH
from gym_paneldepon.state import ACTIONS, NUM_COLORS, State

MAX_CHAIN = 13


class PdPEndlessEnv(gym.Env):
    """
    Panel de Pon environment. Single player endless mode.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, height=HEIGHT, num_colors=NUM_COLORS, max_chain=MAX_CHAIN):
        self.state = State(scoring_method="endless", height=height, num_colors=num_colors)
        self.max_chain = max_chain
        self.reward_range = (0, self.max_chain)
        self.action_space = spaces.Discrete((WIDTH - 1) * self.state.height + 2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.max_chain),
            spaces.Box(0, 1, (self.state.num_colors + 3, self.state.height, WIDTH)),
        ))
        self._seed()

    def _seed(self, seed=None):
        seed = self.state.seed(seed)
        return [seed]

    def _reset(self):
        self.state.reset()
        return (self.state.chain_number, self.state.encode())

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == "ansi" else sys.stdout
        self.state.render(outfile)
        return outfile

    def _step_state(self, state, action, include_observations=True):
        action = ACTIONS[action]
        score = state.step(action)
        reward = min(score, self.max_chain)
        chain_number = min(state.chain_number, self.max_chain - 1)
        if include_observations:
            observation = (chain_number, state.encode())
            return observation, reward
        return reward

    def _step(self, action):
        observation, reward = self._step_state(self.state, action)
        return observation, reward, False, {"state": self.state}

    def get_tree(self, depth=1, include_observations=True):
        """Returns potential observations and rewards up to a search depth"""
        if depth != 1:
            raise NotImplementedError("Only depth 1 trees supported")
        results = []
        for i in range(self.action_space.n):
            clone = self.state.clone()
            results.append(self._step_state(clone, i, include_observations=include_observations))
        if include_observations:
            return results
        return np.array(results, dtype="float")

    def get_root(self):
        clone = self.state.clone()
        clone.seed()
        return clone


def register():
    gym.envs.registration.register(
        id="PdPEndless-v0",
        entry_point="gym_paneldepon.env:PdPEndlessEnv",
        max_episode_steps=200,
        reward_threshold=25.0,
    )
    gym.envs.registration.register(
        id="PdPEndless4-v0",
        entry_point="gym_paneldepon.env:PdPEndlessEnv",
        kwargs={"height": 4, "num_colors": 3, "max_chain": 8},
        max_episode_steps=200,
        reward_threshold=25.0,
    )
