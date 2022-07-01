# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/3/25
@Version : 1.0
"""
import random
import collections
from typing import *
from ..board import *
from .basic import *

__all__ = ["GTPRandomBot"]


class GTPRandomBot(GTPClient):
    name = "random_bot"
    __version__ = "1.0"

    def __init__(self):
        super().__init__()
        self.board = GoBoard()

    def valid_points(self, player: GoPlayer) -> List[GoPoint]:
        return [pos for pos in self.board.valid_points(player) if not self.board.is_point_a_true_eye(pos, player)]

    def counts(self, player: GoPlayer):
        counts = collections.Counter(self.board.grid.flat)
        komi = self.komi if player == GoPlayer.white else -self.komi
        return counts.get(player.value, 0) + komi - counts.get(player.other.value, 0)

    def should_resign(self, player: GoPlayer):
        return self.board.shape[0] + self.counts(player) < 0

    def play(self, player, pos):
        try:
            self.board.play(pos, player)
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
            point = random.choice(points)
            self.board.play(point, player)
            return point

    def boardsize(self, size):
        self.board = GoBoard(size)
        return True

    def clear_board(self):
        self.board.clean()
