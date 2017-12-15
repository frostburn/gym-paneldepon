import pytest

from gym_paneldepon.bitboard import FULL, WIDTH
from gym_paneldepon.state import State

_ = None
R = 0
G = 1
Y = 2
B = 3
P = 4
C = 5
W = 6


def test_raise_stack():
    state = State()
    for i in range(state.height):
        state.raise_stack()
    for i in range(10):
        state.step(None)
    all_panels = 0
    for panels in state.colors:
        all_panels |= panels
    state.render()
    assert (all_panels == FULL)


def test_flat_chain():
    state = State()
    state.colors[0] = 7 << (WIDTH * (state.height - 1))
    state.colors[1] = 6 << (WIDTH * (state.height - 2))
    state.colors[1] |= 8 << (WIDTH * (state.height - 1))
    state.colors[2] = 8 << (WIDTH * (state.height - 2))
    state.colors[2] |= 48 << (WIDTH * (state.height - 1))
    state.sanitize()

    total = 0
    state.render()
    for i in range(5):
        total += state.step(None)[0]
        state.render()
    assert (total == 6)
    assert (not any(state.colors))


def test_time_lag():
    stack = [
        R, _, _, _, _, _,
        R, _, _, _, _, _,
        G, B, B, _, _, _,
        G, B, B, _, _, _,
        G, G, G, _, _, _,
        R, B, B, _, _, _,
    ]
    state = State.from_list(stack)
    total = 0
    for i in range(6):
        state.render()
        total += state.step(None)[0]
    assert (total == 6)
    assert (not any(state.colors))


@pytest.mark.parametrize("time,height", [
    (1, 2),
    (2, 2),
    (1, 1),
    (2, 1),
    (3, 1),
])
def test_insert_support(time, height):
    stack = [
        _, G, _, _, _, _,
        _, R, _, _, _, _,
        B, R, G, G, _, _,
        B, R, R, B, _, _,
    ]
    state = State.from_list(stack)
    total = 0
    print("Insert time={}, height={}".format(time, height))
    for i in range(6):
        state.render()
        action = None
        if i == time:
            action = (state.height - height) * WIDTH
        total += state.step(action)[0]
    assert (total == 3)


@pytest.mark.parametrize("time", [1, 2])
def test_insert_chain(time):
    stack = [
        _, G, _, _, _, _,
        _, R, _, _, _, _,
        G, R, _, _, _, _,
        B, R, _, _, _, _,
        B, G, _, _, _, _,
    ]
    state = State.from_list(stack)
    total = 0
    for i in range(6):
        state.render()
        action = None
        if i == time:
            action = (state.height - 3) * WIDTH
        total += state.step(action)[0]
    assert (total == 3)


def test_late_slip():
    stack = [
        _, R, _, _, _, _,
        R, B, _, _, _, _,
        G, G, R, _, _, _,
        B, G, B, _, _, _,
        G, G, B, _, _, _,
    ]
    state = State.from_list(stack)
    total = 0
    for i in range(8):
        state.render()
        action = None
        if i == 3:
            action = (state.height - 1) * WIDTH
        total += state.step(action)[0]
    assert (total == 6)


def test_side_fall():
    stack = [
        R, _, _, _, _, _,
        G, _, _, _, _, _,
        B, _, R, R, _, _,
        B, G, G, B, G, _,
    ]
    state = State.from_list(stack)
    total = 0
    for i in range(6):
        state.render()
        action = None
        if i == 0:
            action = (state.height - 4) * WIDTH
        elif i == 1:
            action = 3 + (state.height - 1) * WIDTH
        total += state.step(action)[0]
    assert (total == 3)


def test_fall_on():
    stack = [
        B, _, _, _, _, _,
        R, _, _, _, _, _,
        B, _, _, _, _, _,
        B, _, G, _, _, _,
        G, G, _, _, _, _,
    ]
    state = State.from_list(stack)
    total = 0
    for i in range(6):
        state.render()
        action = None
        if i == 0:
            action = (state.height - 4) * WIDTH
        total += state.step(action)[0]
    assert (total == 3)


def test_catch():
    stack = [
        _, G, _, _, _, _,
        _, R, _, _, _, _,
        G, R, _, _, _, _,
        G, R, _, _, _, _,
    ]
    state = State.from_list(stack)
    total = 0
    for i in range(5):
        state.render()
        action = None
        if i == 2:
            action = (state.height - 3) * WIDTH
        total += state.step(action)[0]
    assert (total == 3)


@pytest.mark.parametrize("time", [1, 2])
def test_support_with_fall(time):
    stack = [
        _, G, _, _, _, _,
        G, R, _, _, _, _,
        B, R, G, _, _, _,
        R, R, B, _, _, _,
    ]
    state = State.from_list(stack)
    total = 0
    for i in range(6):
        state.render()
        action = None
        if i == time:
            action = (state.height - 2) * WIDTH
        total += state.step(action)[0]
    assert (total == 3)


def test_no_levitation():
    stack = [
        G, G, _, _, _, _,
        R, R, R, _, _, _,
    ]
    state = State.from_list(stack)
    for i in range(5):
        state.render()
        action = (state.height - 2) * WIDTH
        state.step(action)
    stack = state.to_list()
    assert (stack == [
        _, _, _, _, _, _,
        G, G, _, _, _, _,
    ])


def test_no_flight():
    stack = [
        G, _, _, _, _, _,
        R, _, B, _, _, _,
    ]
    state = State.from_list(stack)
    state.render()
    state.step((state.height - 2) * WIDTH)
    state.render()
    state.step((state.height - 2) * WIDTH + 1)
    for i in range(3):
        state.render()
        state.step(None)
    stack = state.to_list()
    assert (stack == [
        _, _, _, _, _, _,
        R, G, B, _, _, _,
    ])


def test_uneven_from_list():
    bad_stack = [R, G, B]
    with pytest.raises(ValueError):
        State.from_list(bad_stack)


def test_too_big_from_list():
    bad_stack = [R, G, B] * 100
    with pytest.raises(ValueError):
        State.from_list(bad_stack)


def test_nop_raise():
    state = State()
    for i in range(state.height):
        state.raise_stack()
    colors = state.colors[:]
    state.raise_stack()
    assert state.colors == colors


def test_bonus():
    stack = [
        G, _, _, _, _, _,
        R, _, _, _, Y, P,
        R, B, B, Y, P, C,
        R, G, G, B, Y, P,
    ]
    state = State.from_list(stack)
    state.scoring_method = "endless"
    total = 0
    for i in range(10):
        state.render()
        action = None
        if i == 4:
            action = 4 + (state.height - 2) * WIDTH
        if i == 6:
            action = 3 + (state.height - 2) * WIDTH
        total += state.step(action)
    assert (total == 1 + 2 + 2 + 3 + 3)


def test_clone():
    state = State(scoring_method="endless", height=7, num_colors=7)
    for i in range(5):
        state.raise_stack()
    state.step(WIDTH * state.height - 4)
    state.render()

    clone = state.clone()
    assert all(a == b for a, b in zip(clone.colors, state.colors))
    assert clone.falling == state.falling
    assert clone.swapping == state.swapping
    assert clone.chaining == state.chaining
    assert clone.scoring_method == state.scoring_method
    assert clone.height == state.height
    assert clone.num_colors == state.num_colors
