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
    res = input_b == expected_b
    assert np.all(input_b == expected_b)


if __name__ == "__main__":
    pytest.main()
