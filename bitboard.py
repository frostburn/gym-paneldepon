NUM_COLORS = 6
WIDTH = 6
HEIGHT = 12
NUM_BLOCKS = WIDTH * HEIGHT

FULL = sum(1 << i for i in range(NUM_BLOCKS))

TOP = sum(1 << i for i in range(WIDTH))
BOTTOM = TOP << ((HEIGHT - 1) * WIDTH)

LEFT_WALL = sum(1 << i for i in range(0, NUM_BLOCKS, WIDTH))
RIGHT_BLOCK = FULL ^ LEFT_WALL
RIGHT_WALL = LEFT_WALL << (WIDTH - 1)


def print_panels(panels):
    for i in range(HEIGHT):
        for j in range(WIDTH):
            p = 1 << (j + i * WIDTH)
            if panels & p:
                print("@ ", end="")
            else:
                print("* ", end="")
        print()


def panels_to_list(panels):
    return [bool(panels & (1 << i)) for i in range(WIDTH * HEIGHT)]


def panels_from_list(stack):
    panels = 0
    p = 1
    for panel in stack:
        if panel:
            panels |= p
        p <<= 1
    return panels


def left(panels):
    return (panels & RIGHT_BLOCK) >> 1


def right(panels):
    return (panels << 1) & RIGHT_BLOCK


def up(panels):
    return panels >> WIDTH


def down(panels):
    return (panels << WIDTH) & FULL


def beam_up(panels):
    panels |= panels >> WIDTH
    panels |= panels >> (2 * WIDTH)
    panels |= panels >> (4 * WIDTH)
    panels |= panels >> (8 * WIDTH)
    return panels


def get_matches(panels):
    residuals = panels & left(panels) & right(panels)
    horizontal_matches = residuals | left(residuals) | right(residuals)

    residuals = panels & up(panels) & down(panels)
    vertical_matches = residuals | up(residuals) | down(residuals)

    return horizontal_matches | vertical_matches


def drop_one(panels, empty):
    assert (not (panels & empty))
    row = BOTTOM
    for i in range(HEIGHT - 1):
        falling = down(panels) & row & empty
        panels |= falling
        falling = up(falling)
        panels ^= falling
        empty ^= falling
        row = up(row)
    return panels
