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

    def _do_play(self, color: GoPlayer, pos: GoPoint) -> bool:
        try:
            self.board.play(pos, color)
            return True
        except GoIllegalActionError:
            return False

    def valid_points(self, player: GoPlayer) -> List[GoPoint]:
        return [pos for pos in self.board if
                self.board.is_valid_point(pos, player) and not self.board.is_point_a_true_eye(pos, player)]

    def _do_genmove(self, player: GoPlayer) -> Union[GoPoint, str]:
        counts = collections.Counter(self.board.grid.flat)
        komi = self.komi if player == GoPlayer.white else -self.komi
        if counts[GoPlayer.none.value] + counts[player.value] + komi <= counts[player.other.value]:
            return "resign"
        points = self.valid_points(player)
        if len(points) == 0:
            return "pass"
        else:
            point = random.choice(points)
            self.board.play(point, player)
            return point

    def _do_boardsize(self, size: int) -> bool:
        self.board = GoBoard(size)
        return True

    def _do_clear_board(self):
        self.board.clean()
