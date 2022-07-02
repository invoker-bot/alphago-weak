# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/3/25
@Version : 1.0
"""

import random
from typing import *
from .basic import *
from ..board import GoPoint, GoPlayer, GoIllegalActionError, GoBoardAlpha as GoBoard

__all__ = ["GTPRandomBot"]


class GTPRandomBot(GTPClient):
    name = "random_bot"
    __version__ = "1.0"

    def __init__(self, board: GoBoard = None):
        super().__init__()
        self.board = GoBoard() if board is None else board

    def valid_points(self, player: GoPlayer) -> List[GoPoint]:
        return [pos for pos in self.board.valid_points(player) if not self.board.is_point_a_true_eye(pos, player)]

    def should_resign(self, player: GoPlayer):
        return 2 * self.board.size + self.board.score(player, self.komi) < 0

    def play(self, player, pos):
        try:
            self.board.play(player, pos)
            return True
        except GoIllegalActionError:
            return False

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
