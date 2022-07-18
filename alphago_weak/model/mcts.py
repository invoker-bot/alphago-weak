#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/7/13
@Version : 1.0
"""
import math
import copy
import random
import weakref
from typing import *

from alphago_weak.board import *

StateCalculator = Callable[[GoBoardBase, GoPlayer], Any]
PolicyEvaluator = Callable[[Any, float], Tuple[List[GoPoint], List[float]]]
ValueEvaluator = Callable[[Any, float], float]

__all__ = ["GoMCTSNode", "GoMCTSTree"]


class GoMCTSNode:
    c = 1.0
    t = 0.5  # temperature
    depth = 1

    def __init__(self, parent: "GoMCTSNode" = None, pos: GoPoint = None, explore_rate=1.0):
        if parent is None:
            parent = self
        self.parent = weakref.ref(parent)
        self.children: Optional[List["GoMCTSNode"]] = None
        self.w = 0.5
        self.n = 1
        self.p = explore_rate  # policy select probability
        self.pos = pos

    @property
    def win_rate(self) -> float:
        return self.w / self.n

    @property
    def explore_decay(self) -> float:
        N = self.parent().n
        return math.sqrt(N) / self.n

    @property
    def upper_confidence_bound(self) -> float:
        return self.win_rate + self.c * self.p * self.explore_decay

    def evaluate(self, board: GoBoardBase, player: GoPlayer, state_calculator: StateCalculator, policy_evaluator: PolicyEvaluator, value_evaluator: ValueEvaluator, komi=6.5, num=1600) -> Optional[GoPoint]:
        for _ in range(num):
            b_ = copy.deepcopy(board)
            cur_player, node = self.choose(b_, player)
            state = node.expand(b_, cur_player, state_calculator, policy_evaluator, komi)
            node.update(node.rollout(b_, cur_player, state, value_evaluator, komi))
        return self.choose_point()

    def choose(self, board: GoBoardBase, player: GoPlayer) -> Tuple[GoPlayer, "GoMCTSNode"]:
        if self.children is None or len(self.children) == 0:
            return player, self
        max_child = max(self.children, key=lambda node: node.upper_confidence_bound)
        board[max_child.pos] = player
        return max_child.choose(board, player.other)

    def expand(self, board: GoBoardBase, player: GoPlayer, state_calculator: StateCalculator, policy_evaluator: PolicyEvaluator, komi=6.5):
        if self.children is None:
            state = state_calculator(board, player)
            actions, weights = policy_evaluator(state, komi)
            self.children = [self.__class__(self, action, weight) for action, weight in zip(actions, weights)]
            return state
        return None

    def rollout(self, board: GoBoardBase, player: GoPlayer, state: Any, value_evaluator: ValueEvaluator, komi=6.5):
        if len(self.children) == 0:  # at the end of the game
            return float(board.score(player.other, komi) > 0)
        return 1.0 - value_evaluator(state, komi)

    def update(self, w: float = 0):
        """backward propagation"""
        self.n += 1
        self.w += w
        parent = self.parent()
        if parent is not self:
            parent.update(1 - w)

    def choose_point(self) -> Optional[GoPoint]:
        if self.children is None or len(self.children) == 0:
            return None
        points, weights = zip(*map(lambda node: (node.pos, node.n ** (1 / self.t)), self.children))
        return random.choices(points, weights)[0]


class GoMCTSTree(object):

    def __init__(self, board: GoBoardBase, state_calculator: StateCalculator, policy_evaluator: PolicyEvaluator, value_evaluator: ValueEvaluator, player=GoPlayer.black, komi=6.5):
        self.board = board
        self.player = player
        self.komi = komi
        self.node = GoMCTSNode()
        self._state_calculator = state_calculator
        self._policy_evaluator = policy_evaluator
        self._value_evaluator = value_evaluator

    def play(self, player: GoPlayer, pos: GoPoint):
        if self.player == player and self.node.children is not None:
            for child in self.node.children:
                if pos == child.pos:
                    child.parent = weakref.ref(child)
                    self.node = child
                    self.player = player.other
                    return
        self.player = player.other
        self.node = GoMCTSNode()

    def evaluate(self, player, num=1600) -> Optional[GoPoint]:
        assert player == self.player
        return self.node.evaluate(self.board, self.player, self._state_calculator, self._policy_evaluator, self._value_evaluator, self.komi, num)
