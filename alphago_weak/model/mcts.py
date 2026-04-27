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
GoAction = Optional[GoPoint]
PolicyEvaluator = Callable[[Any, float], Tuple[List[GoAction], List[float]]]
ValueEvaluator = Callable[[Any, float], float]
ProgressCallback = Optional[Callable[[int], None]]

__all__ = ["GoMCTSNode", "GoMCTSTree"]


class GoMCTSNode:
    c = 1.0
    t = 0.5  # temperature
    depth = 1

    def __init__(self, parent: "GoMCTSNode" = None, pos: GoAction = None, explore_rate=1.0):
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

    def evaluate(self, board: GoBoardBase, player: GoPlayer, state_calculator: StateCalculator,
                 policy_evaluator: PolicyEvaluator, value_evaluator: ValueEvaluator, komi=6.5,
                 num=1600, progress_callback: ProgressCallback = None,
                 temperature: Optional[float] = None,
                 dirichlet_alpha: Optional[float] = None,
                 exploration_fraction: float = 0.25) -> Optional[GoPoint]:
        noise_applied = False
        if self.children is not None:
            noise_applied = self.add_dirichlet_noise(dirichlet_alpha, exploration_fraction)
        for _ in range(num):
            b_ = copy.deepcopy(board)
            cur_player, node = self.choose(b_, player)
            state = node.expand(b_, cur_player, state_calculator, policy_evaluator, komi)
            if node is self and not noise_applied:
                noise_applied = self.add_dirichlet_noise(dirichlet_alpha, exploration_fraction)
            node.update(node.rollout(b_, cur_player, state, value_evaluator, komi))
            if progress_callback is not None:
                progress_callback(1)
        return self.choose_point(temperature)

    def choose(self, board: GoBoardBase, player: GoPlayer) -> Tuple[GoPlayer, "GoMCTSNode"]:
        if self.children is None or len(self.children) == 0:
            return player, self
        self.children = [
            child for child in self.children
            if child.pos is None or board.is_valid_point(player, child.pos)
        ]
        if len(self.children) == 0:
            return player, self
        max_child = max(self.children, key=lambda node: node.upper_confidence_bound)
        board.play(player, max_child.pos)
        return max_child.choose(board, player.other)

    def expand(self, board: GoBoardBase, player: GoPlayer, state_calculator: StateCalculator, policy_evaluator: PolicyEvaluator, komi=6.5):
        if self.children is None:
            state = state_calculator(board, player)
            actions, weights = policy_evaluator(state, komi)
            self.children = [
                self.__class__(self, action, weight)
                for action, weight in zip(actions, weights)
                if action is None or board.is_valid_point(player, action)
            ]
            self.normalize_child_priors()
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

    def normalize_child_priors(self):
        if self.children is None or len(self.children) == 0:
            return
        for child in self.children:
            if not math.isfinite(child.p) or child.p < 0:
                child.p = 0.0
        total = sum(child.p for child in self.children)
        if total <= 0:
            uniform = 1.0 / len(self.children)
            for child in self.children:
                child.p = uniform
            return
        for child in self.children:
            child.p /= total

    def add_dirichlet_noise(self, alpha: Optional[float], exploration_fraction: float = 0.25,
                            noise: Optional[Sequence[float]] = None) -> bool:
        if alpha is None or alpha <= 0 or exploration_fraction <= 0:
            return False
        if self.children is None or len(self.children) == 0:
            return False
        fraction = min(max(exploration_fraction, 0.0), 1.0)
        if noise is None:
            noise = [random.gammavariate(alpha, 1.0) for _ in self.children]
        if len(noise) != len(self.children):
            raise ValueError("noise length must match child count")
        total_noise = sum(max(float(value), 0.0) for value in noise)
        if total_noise <= 0:
            noise = [1.0 / len(self.children) for _ in self.children]
        else:
            noise = [max(float(value), 0.0) / total_noise for value in noise]
        self.normalize_child_priors()
        for child, noise_value in zip(self.children, noise):
            child.p = (1.0 - fraction) * child.p + fraction * noise_value
        self.normalize_child_priors()
        return True

    def visit_policy(self, temperature: Optional[float] = None) -> Tuple[List[GoAction], List[float]]:
        if self.children is None or len(self.children) == 0:
            return [], []
        if temperature is None:
            temperature = self.t
        if temperature <= 0:
            best_child = max(self.children, key=lambda node: node.n)
            return [child.pos for child in self.children], [1.0 if child is best_child else 0.0 for child in self.children]
        weights = [child.n ** (1 / temperature) for child in self.children]
        total = sum(weights)
        if total <= 0:
            weights = [1.0 / len(self.children) for _ in self.children]
        else:
            weights = [weight / total for weight in weights]
        return [child.pos for child in self.children], weights

    def choose_point(self, temperature: Optional[float] = None) -> Optional[GoPoint]:
        if self.children is None or len(self.children) == 0:
            return None
        points, weights = self.visit_policy(temperature)
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

    def evaluate(self, player, num=1600, progress_callback: ProgressCallback = None,
                 temperature: Optional[float] = None,
                 dirichlet_alpha: Optional[float] = None,
                 exploration_fraction: float = 0.25) -> Optional[GoPoint]:
        assert player == self.player
        return self.node.evaluate(
            self.board,
            self.player,
            self._state_calculator,
            self._policy_evaluator,
            self._value_evaluator,
            self.komi,
            num,
            progress_callback,
            temperature,
            dirichlet_alpha,
            exploration_fraction,
        )
