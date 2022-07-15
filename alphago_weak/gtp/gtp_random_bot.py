# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/3/25
@Version : 1.0
"""

import math
import random
from typing import *
from .basic import *
from ..board import *

__all__ = ["GTPRandomBot"]


class GTPRandomBot(GTPClientBase):
    name = "random_bot"
    __version__ = "1.0"

    def __init__(self, board: GoBoardBase = None, komi=6.5):
        super().__init__(board, komi)
        if self.board is None:
            self.board = GoBoard()

    def valid_points(self, player: GoPlayer) -> List[GoPoint]:
        return [pos for pos in self.board.valid_points(player) if self.board.eye_type(player, pos) < GoEyeType.unknown]

    def should_resign(self, player: GoPlayer):
        return math.log(self.board.size) * self.board.size + self.board.score(player, self.komi) < 0

    def genmove(self, player):
        if self.should_resign(player):
            return "resign"
        points = self.valid_points(player)
        if len(points) == 0:
            return "pass"
        else:
            return random.choice(points)

    def boardsize(self, size):
        self.board = GoBoard(size)
        return True

    def clear_board(self):
        self.board = GoBoard(self.board.size)
