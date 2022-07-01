# -*- coding: utf-8 -*-

import numpy as np
from unittest import TestCase
from typing import *

from ..board import *
from ..dataset import *
from dlgo.gotypes import *
from dlgo.goboard import Board as _Board

black, white, none = GoPlayer.black, GoPlayer.white, GoPlayer.none

__all__ = ["board_check_basic", "board_check_sgf"]


def board_check_basic(case: TestCase, Board: Type[GoBoardBase]):
    b = Board(4)

    def test_grid(grid: List[List[GoPlayer]]):
        case.assertTrue(
            np.all(np.array(grid, np.uint8) == b.grid))

    case.assertTrue(np.all(np.full((4, 4), GoPlayer.none, np.uint8) == b.grid))
    b.play(GoPoint(0, 0), black)
    b.play(GoPoint(0, 1), white)
    b.play(GoPoint(1, 1), black)
    b.play(GoPoint(1, 0), white)
    test_grid([[none, white, none, none],
               [white, black, none, none],
               [none, none, none, none],
               [none, none, none, none]])
    case.assertRaises(GoIllegalActionError, lambda: b.play(GoPoint(0, 0), black))
    for i in range(4):
        for j in range(4):
            if 0 <= i < 2 and 0 <= j < 2:
                case.assertFalse(b.is_valid_point(GoPoint(i, j), black))
            else:
                case.assertTrue(b.is_valid_point(GoPoint(i, j), black))
    test_grid([[none, white, none, none],
               [white, black, none, none],
               [none, none, none, none],
               [none, none, none, none]])
    b.play(GoPoint(2, 0), black)
    b.play(GoPoint(1, 2), white)
    b.play(GoPoint(0, 2), black)
    b.play(GoPoint(3, 3), white)
    b.play(GoPoint(0, 0), black)
    test_grid([[black, none, black, none],
               [none, black, white, none],
               [black, none, none, none],
               [none, none, none, white]])

    b = Board(3)
    case.assertTrue(np.all(np.full((3, 3), GoPlayer.none, np.uint8) == b.grid))
    b.play(GoPoint(0, 1), black)
    b.play(GoPoint(0, 0), white)
    b.play(GoPoint(0, 2), black)
    b.play(GoPoint(2, 0), white)
    b.play(GoPoint(1, 2), black)
    test_grid([[white, black, black],
               [none, none, black],
               [white, none, none]])
    bs = b.get_string(GoPoint(0, 1))
    case.assertTrue(bs == b.get_string(GoPoint(0, 2)) == b.get_string(GoPoint(1, 2)))
    case.assertSetEqual({GoPoint(0, 1), GoPoint(0, 2), GoPoint(1, 2)}, bs.stones)
    case.assertSetEqual({GoPoint(1, 1), GoPoint(2, 2)}, bs.liberties)
    case.assertEqual(GoPlayer.black, bs.player)
    ws1 = b.get_string(GoPoint(0, 0))
    ws2 = b.get_string(GoPoint(2, 0))
    case.assertSetEqual({GoPoint(0, 0)}, ws1.stones)
    case.assertSetEqual({GoPoint(1, 0)}, ws1.liberties)
    case.assertEqual(GoPlayer.white, ws1.player)
    case.assertSetEqual({GoPoint(2, 0)}, ws2.stones)
    case.assertSetEqual({GoPoint(1, 0), GoPoint(2, 1)}, ws2.liberties)
    case.assertEqual(GoPlayer.white, ws2.player)

    b = Board(5)
    b.play(GoPoint(0, 0), black)
    b.play(GoPoint(0, 1), black)
    b.play(GoPoint(0, 2), black)
    b.play(GoPoint(0, 3), black)
    b.play(GoPoint(1, 3), black)
    b.play(GoPoint(2, 0), black)
    b.play(GoPoint(2, 1), black)
    b.play(GoPoint(2, 2), black)
    b.play(GoPoint(2, 3), black)
    b.play(GoPoint(1, 0), white)
    b.play(GoPoint(1, 2), white)

    test_grid([[black, black, black, black, none],
               [white, none, white, black, none],
               [black, black, black, black, none],
               [none, none, none, none, none],
               [none, none, none, none, none]])
    case.assertTrue(b.is_valid_point(GoPoint(1, 1), GoPlayer.black))
    case.assertFalse(b.is_valid_point(GoPoint(1, 1), GoPlayer.white))


def to_point(point: Optional[GoPoint]):
    if point is not None:
        x, y = point
        return Point(x + 1, y + 1)
    return None


def play_to_end(data: GameData, base: _Board):
    b, w, _ = data.setup_stones
    for point in b:
        base.place_stone(Player.black, to_point(point))
    for point in w:
        base.place_stone(Player.white, to_point(point))
    for player, point in data.sequence:
        if point is not None:
            if player == GoPlayer.black:
                base.place_stone(Player.black, to_point(point))
            else:
                base.place_stone(Player.white, to_point(point))


def check_equal(case: TestCase, base: _Board, b: GoBoardBase):
    case.assertEqual((base.num_rows, base.num_cols), b.grid.shape)
    for point in b:
        _player = base.get(to_point(point))
        player = b[point]
        if player == none:
            case.assertIsNone(_player)
        elif player == white:
            case.assertEqual(_player, Player.white)
        else:
            case.assertEqual(_player, Player.black)
    case.assertSetEqual(GoBoardBase.valid_points(b, black), b.valid_points(black))
    case.assertSetEqual(GoBoardBase.valid_points(b, white), b.valid_points(white))


def board_check_sgf(case: TestCase, sgf_name: str, Board: Type[GoBoardBase]):
    data = GameData.from_sgf(sgf_name)
    base = _Board(data.size, data.size)
    play_to_end(data, base)
    b = Board(data.size)
    b.setup_stones(*data.setup_stones)
    for player, point in data.sequence:
        b.play(point, player)
    check_equal(case, base, b)
