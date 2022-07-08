#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import json
import random
from os import path
from glob import iglob
from typing import *
from alphago_weak.board import *
from alphago_weak.dataset import GameData

b, w, none = GoPlayer.black, GoPlayer.white, GoPlayer.none
P = GoPoint


def dump_board_status(_b: GoBoardBase) -> Dict[str, List[Tuple[int, int]]]:
    black_stones: List[Tuple[int, int]] = []
    white_stones: List[Tuple[int, int]] = []
    for pos, player in _b.items():
        if player == GoPlayer.black:
            black_stones.append(tuple(pos))
        elif player == GoPlayer.white:
            white_stones.append(tuple(pos))
    return {"black": black_stones, "white": white_stones}


def assert_board_status(_b: GoBoardBase, status: Dict[str, List[Tuple[int, int]]]):
    board_status = dump_board_status(_b)
    assert set(board_status["black"]) == set(map(tuple, status.get("black", [])))
    assert set(board_status["white"]) == set(map(tuple, status.get("white", [])))


def play_stones(board: GoBoardBase, player: GoPlayer, stones: List[Tuple[int, int]]):
    for stone in stones:
        board.play(player, GoPoint(*stone))
    return stones


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_basic(Board: Type[GoBoardBase]):
    b_ = Board(19)
    # take points and eye judgement
    b_.play(w, P(3, 3))
    b_.play(w, P(3, 4))
    black_stones = play_stones(b_, b, [(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 3), (4, 4), (4, 5)])
    assert_board_status(b_, {"black": black_stones})
    black_stones.extend(play_stones(b_, b, [(3, 4)]))
    assert b_.eye_type(b, P(3, 3)) == GoEyeType.true
    white_stones = play_stones(b_, w, [(1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 6), (3, 1), (3, 6), (4, 1), (4, 6), (5, 2), (5, 3), (5, 4), (5, 5), (3, 3)])
    black_stones.clear()
    assert_board_status(b_, {"white": white_stones, "black": black_stones})

    b_.play(b, P(0, 18))
    b_.play(w, P(0, 17))
    b_.play(w, P(1, 18))
    assert b_.eye_type(w, P(0, 18)) == GoEyeType.unknown
    b_.play(b, P(1, 17))
    black_stones.append((1, 17))
    assert b_.eye_type(w, P(0, 18)) == GoEyeType.fake
    assert not b_.is_valid_point(b, P(0, 18))  # suicide
    black_stones.extend(play_stones(b_, b, [(0, 16), (2, 18)]))
    assert not b_.is_valid_point(w, P(0, 18))  # suicide
    with pytest.raises(GoIllegalActionError):
        b_.play(w, P(0, 18))
    black_stones.extend(play_stones(b_, b, [(0, 18)]))
    assert_board_status(b_, {"white": white_stones, "black": black_stones})

    white_stones.extend(play_stones(b_, w, [(7, 0), (8, 1), (7, 1), (9, 0)]))
    assert b_.eye_type(w, P(8, 0)) == GoEyeType.unknown
    black_stones.extend(play_stones(b_, b, [(9, 1)]))
    assert b_.eye_type(w, P(8, 0)) == GoEyeType.fake
    with pytest.raises(GoIllegalActionError):
        b_.play(b, P(8, 0))
    assert_board_status(b_, {"white": white_stones, "black": black_stones})

    b_.play(w, P(7, 7))
    white_stones.extend(play_stones(b_, w, [(8, 6), (8, 8), (9, 7)]))
    assert b_.eye_type(w, P(8, 7)) == GoEyeType.unknown
    black_stones.extend(play_stones(b_, b, [(7, 6), (7, 8), (6, 7)]))
    assert b_.eye_type(w, P(8, 7)) == GoEyeType.fake
    black_stones.extend(play_stones(b_, b, [(8, 7)]))
    with pytest.raises(GoIllegalActionError):  # robbery
        b_.play(w, P(7, 7))
    assert_board_status(b_, {"white": white_stones, "black": black_stones})
    white_stones.extend(play_stones(b_, w, [(7, 9), (6, 8), (7, 7)]))  # double kill
    black_stones.remove((8, 7))
    black_stones.remove((7, 8))
    assert_board_status(b_, {"white": white_stones, "black": black_stones})


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_random_play(Board: Type[GoBoardBase], benchmark):

    @benchmark
    def _():
        _b = Board()
        current_player = b
        for i in range(_b.size * _b.size):
            valid_points = [pos for pos in _b.valid_points(current_player) if _b.eye_type(current_player, pos) < GoEyeType.unknown]
            if len(valid_points) > 0:
                pos = random.choice(valid_points)
                _b.play(current_player, pos)
            current_player = current_player.other


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_sgf_play(Board: Type[GoBoardBase], benchmark):
    dataset = []
    for sgf_file in iglob(path.join(path.dirname(__file__), "sgf", "**/*.sgf"), recursive=True):
        data = GameData.from_sgf(sgf_file)
        json_data_path = f'{path.splitext(sgf_file)[0]}.json'
        json_data = json.load(open(json_data_path)) if path.isfile(json_data_path) else None
        dataset.append((data, json_data))

    @benchmark
    def _():
        for data, json_data in dataset:
            _b = Board(data.size)
            _b.setup_stones(*data.setup_stones)
            for player, pos in data.sequence:
                _b.play(player, pos)
            if json_data is not None:
                assert_board_status(_b, json_data)


if __name__ == "__main__":
    pytest.main()
