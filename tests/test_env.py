import numpy as np
import pytest
from gym.envs.registration import make

from gym_paneldepon.env import register

register()


@pytest.mark.parametrize("name", ["PdPEndless-v0", "PdPEndless4-v0"])
def test_env(name):
    env = make(name)
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    msg = 'Reset observation: {!r} not in space'.format(ob)
    assert ob_space.contains(ob), msg
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    msg = 'Step observation: {!r} not in space'.format(observation)
    assert ob_space.contains(observation), msg
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)
    env.render(close=True)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)
    env.render(close=True)

    env.close()


@pytest.mark.parametrize("name", ["PdPEndless-v0", "PdPEndless4-v0"])
def test_random_rollout(name):
    env = make(name)
    agent = lambda ob: env.action_space.sample()  # noqa: E731
    ob = env.reset()
    for _ in range(100):
        assert env.observation_space.contains(ob)
        a = agent(ob)
        assert env.action_space.contains(a)
        (ob, _reward, done, _info) = env.step(a)
        env.render(mode="human")
        if done:
            break


@pytest.mark.parametrize("name", ["PdPEndless-v0", "PdPEndless4-v0"])
def test_tree(name):
    env = make(name)
    agent = lambda ob: env.action_space.sample()  # noqa: E731
    observation = env.reset()
    for _ in range(12):
        env.step(1)
    for _ in range(50):
        assert env.observation_space.contains(observation)
        action = agent(observation)
        assert env.action_space.contains(action)
        (observation, reward, done, _info) = env.step(action)
        assert env.reward_range[0] <= reward <= env.reward_range[-1]
        env.render(mode="human")
        rewards = env.unwrapped.get_tree(include_observations=False)
        print(rewards)
        for reward in rewards:
            assert env.reward_range[0] <= reward <= env.reward_range[-1]
        if done:
            break


def test_tree_search():
    env = make("PdPEndless4-v0")

    def deep_agent():
        root = env.unwrapped.get_root()
        best_score = 0
        best_action = np.random.randint(2, env.action_space.n)
        for action, (child, score) in enumerate(root.get_children()):
            for grand_child, child_score in child.get_children():
                for _, grand_child_score in grand_child.get_children():
                    total = score + child_score + grand_child_score
                    if total > best_score:
                        best_action = action
                        best_score = total
        return best_action

    def agent():
        root = env.unwrapped.get_root()
        best_score = 0
        best_action = np.random.randint(2, env.action_space.n)
        for action, (child, score) in enumerate(root.get_children()):
            for grand_child, child_score in child.get_children():
                total = score + child_score
                if total > best_score:
                    best_action = action
                    best_score = total
        return best_action
    env.reset()
    for _ in range(4):
        env.step(1)
    for _ in range(5):
        env.step(agent())
        env.render(mode="human")
