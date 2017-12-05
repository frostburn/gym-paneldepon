from time import sleep

from bitboard import HEIGHT
from state import ACTIONS, RAISE_STACK, State
from util import print_up


def random_demo():
    state = State()
    seed = state.seed()
    print("Random play using seed={}".format(seed))
    for i in range(12):
        state.raise_stack()
    state.render()
    print()
    max_chain = 0
    total = 0
    for i in range(1000):
        if state.np_random.randint(0, 5):
            action = ACTIONS[state.np_random.randint(0, len(ACTIONS))]
        else:
            action = RAISE_STACK
        total += state.step(action)[0]
        sleep(0.1)
        print_up(HEIGHT + 2)
        state.render()
        max_chain = max(state.chain_number, max_chain)
        print("score={} max chain={}".format(total, max_chain))


if __name__ == "__main__":
    random_demo()
