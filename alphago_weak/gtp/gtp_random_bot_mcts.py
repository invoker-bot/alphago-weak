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
from ..board import *
from ..model.mcts import GoMCTSTree
from .gtp_random_bot import *

__all__ = ["GTPRandomBotMCTS"]


class GTPRandomBotMCTS(GTPRandomBot):
    name = "random_bot_plus"
    __version__ = "1.0"

    def __init__(self, size=19, komi=6.5, **_kwargs):
        super().__init__(size, komi)
        self.mcts = GoMCTSTree(self.board, self.state_calculator, self.policy_evaluator, self.value_evaluator, komi=komi)

    @staticmethod
    def state_calculator(b_: GoBoardBase, player: GoPlayer):
        return b_, player

    @staticmethod
    def policy_evaluator(state: Tuple[GoBoardBase, GoPlayer], komi=6.5):
        b_, player = state
        actions = GTPRandomBotMCTS.valid_points(b_, player)
        weights = [1.0 for _ in range(len(actions))]
        return actions, weights

    @staticmethod
    def value_evaluator(state: Tuple[GoBoardBase, GoPlayer], komi=6.5):
        b_, player = state
        cur_player = player
        while True:
            actions = GTPRandomBotMCTS.valid_points(b_, cur_player)
            if len(actions) == 0:
                return float(b_.score(player, komi) > 0)
            pos = random.choice(actions)
            b_[pos] = cur_player
            cur_player = cur_player.other

    def space_count(self):
        count = 0
        for pos in self.board:
            if self.board[pos] == GoPlayer.none:
                count += 1
        return count

    def genmove(self, player: GoPlayer) -> Union[GoPoint, str]:
        if self.should_resign(player):
            return "resign"
        pos = self.mcts.evaluate(num=400)
        if pos is not None:
            return pos
        else:
            return "pass"

    def play(self, player, pos):
        result = super().play(player, pos)
        self.mcts.play(player, pos)
        return result

    def clear_board(self):
        super().clear_board()
        self.mcts = GoMCTSTree(self.board, self.state_calculator, self.policy_evaluator, self.value_evaluator, komi=self.komi)
