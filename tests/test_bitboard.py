from six import StringIO

from gym_paneldepon import bitboard


def test_print():
    outfile = StringIO()
    bitboard.print_panels(12345, outfile=outfile)
    result = (
        '@ * * @ @ @ \n'
        '* * * * * * \n'
        '@ @ * * * * \n'
        '* * * * * * \n'
        '* * * * * * \n'
        '* * * * * * \n'
        '* * * * * * \n'
        '* * * * * * \n'
        '* * * * * * \n'
        '* * * * * * \n'
        '* * * * * * \n'
        '* * * * * * \n'
    )
    assert outfile.getvalue() == result


def test_beam():
    assert bitboard.beam_up(bitboard.BOTTOM) == bitboard.FULL


def test_to_list():
    result = [
        1, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    ]
    assert list(map(int, bitboard.panels_to_list(12345))) == result
