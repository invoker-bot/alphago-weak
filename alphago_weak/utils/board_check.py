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


def board_check_basic(case: TestCase, Board: Type[GoBoardProtocol]):
    gb = Board(4)

    def test_grid(grid: List[List[GoPlayer]]):
        for pos in gb:
            color = gb[pos]
            case.assertEqual(grid[pos.x][pos.y], color)
    gb.play(black, GoPoint(0, 0))
    gb.play(white, GoPoint(0, 1))
    gb.play(black, GoPoint(1, 1))
    gb.play(white, GoPoint(1, 0))
    test_grid([[none, white, none, none],
               [white, black, none, none],
               [none, none, none, none],
               [none, none, none, none]])
    case.assertRaises(GoIllegalActionError, lambda: gb.play(black, GoPoint(0, 0)))
    for i in range(4):
        for j in range(4):
            if 0 <= i < 2 and 0 <= j < 2:
                case.assertFalse(gb.is_valid_point(black, GoPoint(i, j)))
            else:
                case.assertTrue(gb.is_valid_point(black, GoPoint(i, j)))
    test_grid([[none, white, none, none],
               [white, black, none, none],
               [none, none, none, none],
               [none, none, none, none]])
    gb.play(black, GoPoint(2, 0))
    gb.play(white, GoPoint(1, 2))
    gb.play(black, GoPoint(0, 2))
    gb.play(white, GoPoint(3, 3))
    gb.play(black, GoPoint(0, 0))
    test_grid([[black, none, black, none],
               [none, black, white, none],
               [black, none, none, none],
               [none, none, none, white]])

    gb = Board(3)
    gb.play(black, GoPoint(0, 1))
    gb.play(white, GoPoint(0, 0))
    gb.play(black, GoPoint(0, 2))
    gb.play(white, GoPoint(2, 0))
    gb.play(black, GoPoint(1, 2))
    test_grid([[white, black, black],
               [none, none, black],
               [white, none, none]])
    bs = gb.get_string(GoPoint(0, 1))
    case.assertTrue(bs == gb.get_string(GoPoint(0, 2)) == gb.get_string(GoPoint(1, 2)))
    case.assertSetEqual({GoPoint(0, 1), GoPoint(0, 2), GoPoint(1, 2)}, bs.stones)
    case.assertSetEqual({GoPoint(1, 1), GoPoint(2, 2)}, bs.liberties)
    case.assertEqual(GoPlayer.black, bs.player)
    ws1 = gb.get_string(GoPoint(0, 0))
    ws2 = gb.get_string(GoPoint(2, 0))
    case.assertSetEqual({GoPoint(0, 0)}, ws1.stones)
    case.assertSetEqual({GoPoint(1, 0)}, ws1.liberties)
    case.assertEqual(GoPlayer.white, ws1.player)
    case.assertSetEqual({GoPoint(2, 0)}, ws2.stones)
    case.assertSetEqual({GoPoint(1, 0), GoPoint(2, 1)}, ws2.liberties)
    case.assertEqual(GoPlayer.white, ws2.player)

    gb = Board(5)
    gb.play(black, GoPoint(0, 0))
    gb.play(black, GoPoint(0, 1))
    gb.play(black, GoPoint(0, 2))
    gb.play(black, GoPoint(0, 3))
    gb.play(black, GoPoint(1, 3))
    gb.play(black, GoPoint(2, 0))
    gb.play(black, GoPoint(2, 1))
    gb.play(black, GoPoint(2, 2))
    gb.play(black, GoPoint(2, 3))
    gb.play(white, GoPoint(1, 0))
    gb.play(white, GoPoint(1, 2))

    test_grid([[black, black, black, black, none],
               [white, none, white, black, none],
               [black, black, black, black, none],
               [none, none, none, none, none],
               [none, none, none, none, none]])
    case.assertTrue(gb.is_valid_point(black, GoPoint(1, 1)))
    case.assertFalse(gb.is_valid_point(white, GoPoint(1, 1)))


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
    case.assertEqual((base.num_rows, base.num_cols), (b.size, b.size))
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
        b.play(player, point)
    check_equal(case, base, b)
