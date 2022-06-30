# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2021/3/25
@Version : 1.0
"""

from typing import *

from ..basic import *
from .gtp_random_bot import *

__all__ = ["GTPRandomBotPlus"]


class GTPRandomBotPlus(GTPRandomBot):
    name = "random_bot_plus"

    def valid_points(self, player: GoPlayer) -> List[GoPoint]:
        points = []
        for pos in self.board:
            if self.board.is_valid_point(pos, player):
                other_counts = 0
                for neighbor in self.board.get_neighbors(pos):
                    if self.board.grid.item(neighbor) != self.board.next_player.value:
                        other_counts += 1
                if other_counts > 0:
                    points.append(pos)
                    continue
                for corner in self.board.get_corners(pos):
                    if self.board.grid.item(corner) != self.board.next_player.value:
                        other_counts += 1
                if other_counts > 1:
                    points.append(pos)
        return points
