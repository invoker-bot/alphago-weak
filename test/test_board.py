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


def test_go_point_is_type_safe_value_object():
    point = P(1, 2)

    assert GoPoint.to_point(point) is point
    assert P(point, 3) == P(1, 3)
    assert point != (1, 2)
    with pytest.raises(TypeError):
        P("1", 2)


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_rejects_out_of_bounds_points(Board: Type[GoBoardBase]):
    b_ = Board(5)

    for pos in (P(-1, 0), P(0, -1), P(5, 0), P(0, 5)):
        assert not b_.is_valid_point(b, pos)
        with pytest.raises(GoIllegalActionError):
            b_.play(b, pos)


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_area_score_counts_connected_territory(Board: Type[GoBoardBase]):
    b_ = Board(5)
    for x in range(5):
        for y in range(5):
            if x in (0, 4) or y in (0, 4):
                b_.play(b, P(x, y))

    assert b_.score(b, 0) == 25
    assert b_.score(w, 0) == -25
    assert b_.score(b, 6.5) == 18.5
    assert b_.score(w, 6.5) == -18.5


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_setup_stones_places_static_position_without_captures(Board: Type[GoBoardBase]):
    b_ = Board(3)
    black_stones = {P(0, 0), P(0, 1), P(0, 2), P(1, 0), P(1, 2), P(2, 0), P(2, 1), P(2, 2)}
    white_stones = {P(1, 1)}

    b_.setup_stones(black_stones, white_stones, set())

    assert_board_status(
        b_,
        {
            "black": list(map(tuple, black_stones)),
            "white": [(1, 1)],
        },
    )


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_copy_preserves_position_and_is_independent(Board: Type[GoBoardBase]):
    b_ = Board(5)
    b_.play(b, P(0, 0))
    b_.play(w, P(1, 0))
    b_.play(b, P(0, 1))
    b_.play(w, P(2, 2))
    b_.play(b, P(4, 4))

    copied = b_.copy()

    assert type(copied) is Board
    assert copied.size == b_.size
    assert hash(copied) == hash(b_)
    assert_board_status(copied, dump_board_status(b_))

    copied.play(w, P(3, 4))

    assert copied[P(3, 4)] == w
    assert b_[P(3, 4)] == none
    assert hash(copied) != hash(b_)


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_valid_points_match_point_legality(Board: Type[GoBoardBase]):
    b_ = Board(5)
    for player, pos in (
        (b, P(0, 0)),
        (w, P(1, 0)),
        (b, P(0, 1)),
        (w, P(2, 2)),
        (b, P(4, 4)),
    ):
        b_.play(player, pos)

    valid_from_iterator = set(b_.valid_points(w))
    valid_from_points = {pos for pos in b_ if b_.is_valid_point(w, pos)}

    assert valid_from_iterator == valid_from_points


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
    assert b_.score(b, 6.5) == -36.5
    assert b_.score(w, 6.5) == 36.5


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_random_play(Board: Type[GoBoardBase], benchmark):

    def run_random_game():
        rng = random.Random(0)
        _b = Board()
        current_player = b
        for i in range(_b.size * _b.size):
            valid_points = [pos for pos in _b.valid_points(current_player) if _b.eye_type(current_player, pos) < GoEyeType.unknown]
            if len(valid_points) > 0:
                pos = rng.choice(valid_points)
                _b.play(current_player, pos)
            current_player = current_player.other

    benchmark.pedantic(run_random_game, rounds=1, iterations=1)


@pytest.mark.parametrize("Board", [GoBoardAlpha, GoBoardBeta])
def test_board_sgf_play(Board: Type[GoBoardBase], benchmark):
    dataset = []
    for sgf_file in iglob(path.join(path.dirname(__file__), "sgf", "**/*.sgf"), recursive=True):
        data = GameData.from_sgf(sgf_file)
        json_data_path = f'{path.splitext(sgf_file)[0]}.json'
        json_data = json.load(open(json_data_path)) if path.isfile(json_data_path) else None
        dataset.append((data, json_data))

    def replay_sgf_games():
        for data, json_data in dataset:
            _b = Board(data.size)
            _b.setup_stones(*data.setup_stones)
            for player, pos in data.sequence:
                _b.play(player, pos)
            if json_data is not None:
                assert_board_status(_b, json_data)

    benchmark.pedantic(replay_sgf_games, rounds=1, iterations=1)


if __name__ == "__main__":
    pytest.main()
