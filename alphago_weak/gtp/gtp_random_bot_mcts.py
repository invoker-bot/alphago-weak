# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2021/3/25
@Version : 1.0
"""

import math
import copy
from typing import *

from ..board import *
from .gtp_random_bot import *

__all__ = ["GTPRandomBotMCTS"]


class MCTSNode(object):

    def __init__(self, parent: "MCTSNode" = None, action: Any = None):
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.action = action
        self.w = 0
        self.n = 0
        self.c = math.sqrt(2)

    @property
    def upper_confidence_bound(self):
        if self.n == 0:
            return 3.0
        N = self.parent.n
        return self.w / self.n + self.c * math.sqrt(math.log(N) / self.n)

    def select_node(self):
        return max(self.children, key=lambda node: node.upper_confidence_bound, default=self)

    def expand(self, actions: list):
        for action in actions:
            self.children.append(MCTSNode(self, action))

    def backward_propagation(self, win: int = 0):
        node = self
        while node is not None:
            node.n += 1
            node.w += win
            node = node.parent


class GTPRandomBotMCTS(GTPRandomBot):
    name = "random_bot_plus"
    __version__ = "1.0"

    def genmove(self, player: GoPlayer) -> Union[GoPoint, str]:
        if self.should_resign(player):
            return "resign"
        points = self.valid_points(player)
        if len(points) == 0:
            return "pass"
        node = MCTSNode()
        node.expand(points)
        board = self.board
        for i in range(self.config.get("mtcs", 1600)):
            self.board = copy.deepcopy(board)
            child_node = node.select_node()
            action = child_node.action
            self.play(player, action)
            current_player = player.other
            while isinstance(action, GoPoint):
                action = super().genmove(current_player)
                current_player = current_player.other
            win = int(self.counts(player) > 0)
            child_node.backward_propagation(win)
        action = node.select_node().action
        self.board = board
        if action is not None:
            return action
        else:
            return "pass"
