from unittest import TestCase
from gotypes import *
from typing import *
from godata import *

black, white, none = GoPlayer.black, GoPlayer.white, GoPlayer.none


def check_v0(case: TestCase, Board: Type[GoBoardBase]):
    b = Board(4)

    def test_grid(grid: List[List[GoPlayer]]):
        case.assertTrue(
            np.all(np.array(grid, np.uint8) == b.grid))

    case.assertTrue(np.all(np.full((4, 4), GoPlayer.none,np.uint8) == b.grid))
    b.play((0, 0))
    b.play((0, 1))
    b.play((1, 1))
    b.play((1, 0))
    test_grid([[none, white, none, none],
               [white, black, none, none],
               [none, none, none, none],
               [none, none, none, none]])
    case.assertRaises(GoIllegalActionError, lambda: b.play((0, 0)))
    for i in range(4):
        for j in range(4):
            if 0 <= i < 2 and 0 <= j < 2:
                case.assertFalse(b.is_valid_point((i, j)))
            else:
                case.assertTrue(b.is_valid_point((i, j)))
    test_grid([[none, white, none, none],
               [white, black, none, none],
               [none, none, none, none],
               [none, none, none, none]])
    b.play((2, 0))
    b.play((1, 2))
    b.play((0, 2))
    b.play((3, 3))
    b.play((0, 0))
    test_grid([[black, none, black, none],
               [none, black, white, none],
               [black, none, none, none],
               [none, none, none, white]])

    b = Board(3)
    case.assertTrue(np.all(np.full((3, 3), GoPlayer.none,np.uint8) == b.grid))
    b.play((0, 1))
    b.play((0, 0))
    b.play((0, 2))
    b.play((2, 0))
    b.play((1, 2))
    test_grid([[white, black, black],
               [none, none, black],
               [white, none, none]])
    bs = b.get_string((0, 1))
    case.assertTrue(bs == b.get_string((0, 2)) == b.get_string((1, 2)))
    case.assertSetEqual({(0, 1), (0, 2), (1, 2)}, bs.stones)
    case.assertSetEqual({(1, 1), (2, 2)}, bs.liberties)
    case.assertEqual(GoPlayer.black, bs.player)
    ws1 = b.get_string((0, 0))
    ws2 = b.get_string((2, 0))
    case.assertSetEqual({(0, 0)}, ws1.stones)
    case.assertSetEqual({(1, 0)}, ws1.liberties)
    case.assertEqual(GoPlayer.white, ws1.player)
    case.assertSetEqual({(2, 0)}, ws2.stones)
    case.assertSetEqual({(1, 0), (2, 1)}, ws2.liberties)
    case.assertEqual(GoPlayer.white, ws2.player)


def check_sample_v0(case: TestCase, Board: Type[GoBoardBase]):
    with SgfDataBase() as data:
        games = data.sample(5)
        for sgf_file in games:
            b = Board(19,sgf_file.first_player)
            b.setup_stones(*sgf_file.setup_stones)
            for pos in sgf_file.sequence:
                b.play(pos)


