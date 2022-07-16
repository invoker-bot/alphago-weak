# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2021/3/25
@Version : 1.0
"""
import sys
import math
import copy
from typing import *

from ..board import *
from ..model.mcts import MCTSNode
from .gtp_random_bot import *

__all__ = ["GTPRandomBotMCTS"]


class GTPRandomBotMCTS(GTPRandomBot):
    name = "random_bot_plus"
    __version__ = "1.0"

    def __init__(self, board: GoBoardBase = None, komi=6.5):
        super().__init__(board, komi)

    @staticmethod
    def policy_evaluator(b_: GoBoardBase, player: GoPlayer):
        actions = [pos for pos in b_.valid_points(player) if b_.eye_type(player, pos) < GoEyeType.unknown]
        weights = [1.0 for _ in range(len(actions))]
        return actions, weights

    def space_count(self):
        count = 0
        for pos in self.board:
            if self.board[pos] == GoPlayer.none:
                count += 1
        return count

    def genmove(self, player: GoPlayer) -> Union[GoPoint, str]:
        if self.should_resign(player):
            return "resign"
        if self.space_count() > 2/3 * (self.board.size**2):
            return super().genmove(player)
        mcts = MCTSNode()
        point = mcts.evaluate(self.board, player, self.policy_evaluator, num=400, komi=self.komi)
        if point is not None:
            return point
        else:
            return "pass"
