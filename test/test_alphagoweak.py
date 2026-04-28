#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/7/12
@Version : 1.0
"""

import pytest
import numpy as np
from alphago_weak.board import *
from alphago_weak.model import *

b, w, none = GoPlayer.black, GoPlayer.white, GoPlayer.none
P = GoPoint


def test_encode_v0():
    b_ = GoBoard(7)
    b_.play(b, P(1, 1))
    b_.play(w, P(2, 3))
    expected_b = np.zeros((11, 7, 7), dtype=np.float32)
    expected_b[0, 1, 1] = 1.0
    expected_b[1, 2, 3] = 1.0
    expected_b[2] = np.ones((7, 7), dtype=np.float32)
    expected_b[2, 1, 1] = 0.0
    expected_b[2, 2, 3] = 0.0
    expected_b[6, 1, 1] = 1.0
    expected_b[10, 2, 3] = 1.0
    input_b = AlphaGoWeakV0.encode_input(b, b_)
    assert np.all(input_b == expected_b)


def test_predict_batch_matches_single_prediction(tmp_path):
    model = AlphaGoWeakV0(size=3, root=str(tmp_path), num_filters=4, device="cpu")
    board = GoBoard(3)
    board.play(b, P(0, 0))

    single_policy, single_value = model.predict(board, w)
    batch = model.predict_batch([(board, w), (board, b)])

    assert len(batch) == 2
    assert np.allclose(batch[0][0], single_policy)
    assert batch[0][1] == single_value
    assert batch[1][0].shape == (10,)


if __name__ == "__main__":
    pytest.main()
