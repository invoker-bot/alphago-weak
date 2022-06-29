# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/3/26
@Version : 1.0
"""

import numpy as np
from typing import *

from ..basic import *
from ..layers import *
from .gtp import *

__all__ = ["GTPAlphaGoWeakV0"]


class GTPAlphaGoWeakV0(GTPClient):
    name = "alphago_weak_v0"
    __version__ = "0.0"

    def __init__(self, size=19, weights_file: str = "default.h5"):
        super().__init__()
        self.size = size
        self.model = AlphaGoWeakV0(size)
        self.model.load(weights_file)
        self.board = GoBoardAIV0(size)

    def _do_play(self, color: GoPlayer, pos: GoPoint) -> bool:
        try:
            self.board.play(pos, color)
            return True
        except GoIllegalActionError:
            return False

    def _do_genmove(self, color: GoPlayer) -> Union[GoPoint, str]:
        return self.model.predict(self.board, color)

    def _do_boardsize(self, size: int) -> bool:
        self.board = GoBoardAIV0(size)
        return True

    def _do_clear_board(self) -> NoReturn:
        self.board = GoBoardAIV0(self.board.grid.shape[0])
