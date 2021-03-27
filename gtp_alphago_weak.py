#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/3/26
@Version : 1.0
"""

import numpy as np
import collections
from gtp import *
from typing import *
from go_types import *
from alpha_go_weak import *

__all__ = ["GTPAlphaGoWeakV0"]


class GTPAlphaGoWeakV0(GTPClient):
    name = "alphago_weak_v0"
    __version__ = "0.0"

    def __init__(self, size=19, weights_file: str = "default.h5"):
        super().__init__()
        self.size = size
        self.model = AlphaGoWeak(size)
        self.model.load(weights_file)
        self.board = GoBoardAI(size)

    def _do_play(self, color: GoPlayer, pos: GoPoint) -> bool:
        try:
            self.board._next_player = color
            self.board.play(pos, color)
            return True
        except GoIllegalActionError:
            return False

    def _do_genmove(self, color: GoPlayer) -> Union[GoPoint, str]:
        self._next_player = color
        weights = self.model.policy_predict(self.board)
        # print(weights)
        num = self.size * self.size
        weights **= 3
        eps = 1e-6
        np.clip(weights, eps, 1 - eps, weights)
        weights /= np.sum(weights)
        index = np.random.choice(np.arange(num), size=num, replace=False, p=weights)
        for i in index:
            pos = divmod(i, self.size)
            if self.board.is_valid_point(pos) and not self.board.is_point_a_true_eye(pos):
                return pos
        return "pass"

    def _do_boardsize(self, size: int) -> bool:
        self.board = GoBoardAI(size)
        return True

    def _do_clear_board(self) -> NoReturn:
        self.board = GoBoardAI(self.board.grid.shape[0])


if __name__ == '__main__':
    client = GTPAlphaGoWeakV0()
    client.mainloop()
