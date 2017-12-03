def print_color(color, bright=True, end=""):
    if bright:
        print("\x1b[3{};1m".format(color), end=end)
    else:
        print("\x1b[3{}m".format(color), end=end)


def print_reset(end=""):
    print("\x1b[0m", end="")


def print_up(n):
    for _ in range(n):
        print("\033[A", end="")
