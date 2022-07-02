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
from .gtp_random_bot import *

__all__ = ["GTPRandomBotMCTS"]


class RandomMCTSNode(object):

    def __init__(self, parent: "RandomMCTSNode" = None, board: GoBoardBase = None, player: GoPlayer = GoPlayer.none,
                 point: GoPoint = None, komi=6.5):
        self.parent = parent
        if board is None:
            board = self.parent.board
        self.board = copy.deepcopy(board)
        self.children: Optional[List["RandomMCTSNode"]] = None
        self.player = parent.player.other if player == GoPlayer.none else player
        self.point = point
        if point is not None:
            self.board.play(self.player, point)
        self.komi = komi
        self.w = 0
        self.n = 0
        self.c = math.sqrt(2)

    @property
    def upper_confidence_bound(self):
        if self.n == 0:
            return sys.float_info.max
        N = self.parent.n
        return self.w / self.n + self.c * math.sqrt(math.log(N) / self.n)

    def evaluate(self, num=1600):
        if self.children is None:
            self.expand()
        for _ in range(num):
            child_node = self.choose_node()
            w = child_node.rollout()
            child_node.update(w)

    def choose_node(self):
        return max(self.children, key=lambda node: node.upper_confidence_bound, default=self)

    def choose_point(self):
        return max(self.children, key=lambda node: node.n, default=self).point

    def expand(self):
        self.children = [RandomMCTSNode(self, point=point, komi=self.komi) for point in self.board.valid_points(self.player.other)]

    def rollout(self):
        board = copy.deepcopy(self.board)
        bot = GTPRandomBot(board)
        player = self.player.other
        while True:
            pos = bot.genmove(player)
            if isinstance(pos, GoPoint):
                assert bot.play(player, pos)
            else:
                break
            player = player.other
        return int(board.score(self.player, self.komi) > 0)

    def update(self, win: int = 0):
        """backward propagation"""
        node = self
        while node is not None:
            node.n += 1
            node.w += win
            win = -win
            node = node.parent


class GTPRandomBotMCTS(GTPRandomBot):
    name = "random_bot_plus"
    __version__ = "1.0"

    def genmove(self, player: GoPlayer) -> Union[GoPoint, str]:
        if self.should_resign(player):
            return "resign"
        node = RandomMCTSNode(None, self.board, player.other, komi=self.komi)
        node.evaluate(self.config.get("mtcs", 1600))
        point = node.choose_point()
        if point is not None:
            return point
        else:
            return "pass"
