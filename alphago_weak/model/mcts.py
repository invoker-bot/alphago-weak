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
from collections import defaultdict
from typing import *

from alphago_weak.board import *

PolicyEvaluator = Callable[[GoBoardBase, GoPlayer], Tuple[List[GoPoint], List[float]]]
ValueEvaluator = Callable[[GoBoardBase, GoPlayer, float], float]

__all__ = ["MCTSNode", "MCTSTree"]


def value_evaluator_wrapper(policy_evaluator: PolicyEvaluator, value_evaluator: ValueEvaluator = None, depth: int = None) -> ValueEvaluator:

    if value_evaluator is not None:
        return value_evaluator

    def evaluator(board: GoBoardBase, player: GoPlayer, komi=6.5) -> float:
        depth_: int = board.size * board.size if depth is None else depth
        current_player = player
        for _ in range(depth_):
            actions, weights = policy_evaluator(board, current_player)
            if len(actions) == 0:  # end of the game
                return float(board.score(player, komi) > 0)
            pos = random.choices(actions, weights)[0]
            board[pos] = current_player  # change board and player
            current_player = current_player.other
        if value_evaluator is None:
            return float(board.score(player, komi) > 0)
        score = value_evaluator(board, current_player, komi)
        if current_player != player:
            score = 1.0 - score
        return score
    return evaluator


class MCTSNode:
    c = 1.0
    t = 1.0  # temperature
    depth = 0.5

    def __init__(self, parent: "MCTSNode" = None, explore_rate=1.0):
        if parent is None:
            parent = self
        self.parent = weakref.ref(parent)
        self.children: Optional[Dict[GoPoint, "MCTSNode"]] = None
        self.w = 0.5
        self.n = 1
        self.p = explore_rate  # policy select probability

    @property
    def win_rate(self) -> float:
        return self.w / self.n

    @property
    def explore_decay(self) -> float:
        N = self.parent().n
        return math.sqrt(math.log(N) / self.n)

    @property
    def upper_confidence_bound(self) -> float:
        return self.win_rate + self.c * self.p * self.explore_decay

    def evaluate(self, board: GoBoardBase, player: GoPlayer, policy_evaluator: PolicyEvaluator, value_evaluator: ValueEvaluator = None, komi=6.5, num=1600) -> Optional[GoPoint]:
        for _ in range(num):
            b_ = copy.deepcopy(board)
            if self.rollout(b_, player, policy_evaluator, value_evaluator_wrapper(policy_evaluator, value_evaluator, self.depth), komi):
                break
        return self.choose_point()

    def choose_node(self) -> Tuple[GoPoint, "MCTSNode"]:
        max_node: Any = max(self.children.items(), key=lambda pair: pair[1].upper_confidence_bound)
        return max_node

    def choose_point(self) -> Optional[GoPoint]:
        if self.children is None or len(self.children) == 0:
            return None
        points = list(self.children.keys())
        weights = [child_node.n ** (1 / self.t) for child_node in self.children.values()]
        return random.choices(points, weights)[0]

    def rollout(self, board: GoBoardBase, player: GoPlayer, policy_evaluator: PolicyEvaluator, value_evaluator: ValueEvaluator, komi=6.5) -> bool:
        if self.children is None:
            actions, weights = policy_evaluator(board, player)
            if len(actions) == 0:  # at the end of the game
                self.update(float(board.score(player.other, komi) > 0))
                return True
            self.children = dict()
            for action, weight in zip(actions, weights):
                self.children[action] = self.__class__(self, weight)
        pos, child_node = self.choose_node()
        board[pos] = player  # change board and player
        player = player.other
        child_node.update(1.0 - value_evaluator(board, player, komi))
        return False

    def update(self, w: float = 0):
        """backward propagation"""
        node = self
        while True:
            node.n += 1
            node.w += w
            w = -w
            parent = node.parent()
            if parent is node:
                break
            node = parent


class MCTSTree(object):

    def __init__(self, ):
        self.cache = defaultdict(lambda: MCTSNode())
