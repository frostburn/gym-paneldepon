import sys

import gym
import gym.envs.registration
from gym import spaces
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

    def _step(self, action):
        action = ACTIONS[action]
        score = self.state.step(action)
        reward = min(score, self.max_chain)
        chain_number = min(self.state.chain_number, self.max_chain - 1)
        observation = (chain_number, self.state.encode())
        return observation, reward, False, {"state": self.state}


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
