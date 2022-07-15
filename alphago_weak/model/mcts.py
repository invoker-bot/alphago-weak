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
from collections import defaultdict
from abc import *
from typing import *

from alphago_weak.board import *

PolicyEvaluator = Callable[[GoBoardBase, GoPlayer], Tuple[List[GoPoint], List[float]]]
ValueEvaluator = Callable[[GoBoardBase, GoPlayer, float], float]


def value_evaluator_wrapper(policy_evaluator: PolicyEvaluator, value_evaluator: ValueEvaluator = None) -> ValueEvaluator:
    if value_evaluator is not None:
        return value_evaluator

    def evaluator(board: GoBoardBase, player: GoPlayer, komi=6.5) -> float:
        current_player = player
        while True:
            actions, weights = policy_evaluator(board, current_player)
            if len(actions) == 0:  # end of the game
                return float(board.score(player, komi) > 0)
            pos = random.choices(actions, weights)[0]
            board[pos] = current_player  # change board and player
            current_player = current_player.other

    return evaluator


class MCTSNode(metaclass=ABCMeta):
    c = 1.0
    t = 1.0  # temperature
    depth = 1

    def __init__(self, parent: "MCTSNode" = None, explore_rate=1.0):
        self.parent = parent
        self.children: Optional[DefaultDict[GoPoint, "MCTSNode"]] = None
        self.w = 0.0
        self.n = 0
        self.p = explore_rate  # policy select probability

    @property
    def win_rate(self) -> float:
        if self.n == 0:
            return 1.0
        return self.w / self.n

    @property
    def explore_decay(self) -> float:
        N = self.parent.n
        return math.sqrt(math.log(N) / (self.n + 1))

    @property
    def upper_confidence_bound(self) -> float:
        return self.win_rate + self.c * self.p * self.explore_decay

    def evaluate(self, board: GoBoardBase, player: GoPlayer, policy_evaluator: PolicyEvaluator, value_evaluator: ValueEvaluator = None, komi=6.5, num=1600) -> Optional[GoPoint]:
        for _ in range(num):
            b_ = copy.deepcopy(board)
            self.rollout(b_, player, policy_evaluator, value_evaluator, komi, self.depth)
        if self.children is not None:
            pos = self.choose_point()
            return pos
        return None

    def next(self, pos: GoPoint = None):
        if pos is None:
            return self
        node = self.children[pos]
        node.parent = None
        return node

    def choose_node(self) -> Tuple[GoPoint, "MCTSNode"]:
        return max(self.children.items(), key=lambda pair: pair[1].upper_confidence_bound)

    def choose_point(self) -> GoPoint:
        points = list(self.children.keys())
        weights = [child_node.n ** (1 / self.t) for child_node in self.children.values()]
        return random.choices(points, weights)[0]

    def rollout(self, board: GoBoardBase, player: GoPlayer, policy_evaluator: Callable[[GoBoardBase, GoPlayer], Tuple[List[GoPoint], List[float]]], value_evaluator: Callable[[GoBoardBase, GoPlayer], float] = None, komi=6.5, depth=5):
        cur_node = self
        for _ in range(depth):
            if cur_node.children is None:
                actions, weights = policy_evaluator(board, player)
                if len(actions) == 0:  # end of the game
                    return cur_node.update(float(board.score(player.other, komi) > 0))
                cur_node.children = defaultdict(lambda: self.__class__(self))
                pos = random.choices(actions, weights)[0]
                cur_node = cur_node.children[pos]
            else:
                pos, cur_node = cur_node.choose_node()
            board[pos] = player  # change board and player
            player = player.other
        return cur_node.update(1.0 - value_evaluator_wrapper(policy_evaluator, value_evaluator)(board, player, komi))

    def update(self, w: float = 0):
        """backward propagation"""
        node = self
        while node is not None:
            node.n += 1
            node.w += w
            w = -w
            node = node.parent
