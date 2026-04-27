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
        actions.append(None)
        weights = [1.0 for _ in range(len(actions))]
        return actions, weights

    @staticmethod
    def value_evaluator(state: Tuple[GoBoardBase, GoPlayer], komi=6.5):
        b_, player = state
        cur_player = player
        consecutive_passes = 0
        while True:
            actions = GTPRandomBotMCTS.valid_points(b_, cur_player)
            random.shuffle(actions)
            played = False
            for pos in actions:
                try:
                    b_.play(cur_player, pos)
                except GoIllegalActionError:
                    continue
                played = True
                break
            if played:
                consecutive_passes = 0
            else:
                b_.play(cur_player, None)
                consecutive_passes += 1
            if consecutive_passes >= 2:
                return float(b_.score(player, komi) > 0)
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
        pos = self.mcts.evaluate(player, num=400)
        if pos is not None:
            return pos
        else:
            return "pass"

    def play(self, player, pos):
        result = super().play(player, pos)
        if result:
            self.mcts.play(player, pos)
        return result

    def clear_board(self):
        super().clear_board()
        self.mcts = GoMCTSTree(self.board, self.state_calculator, self.policy_evaluator, self.value_evaluator, komi=self.komi)
