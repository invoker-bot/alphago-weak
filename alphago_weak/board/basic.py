# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/11/19
@Version : 1.0
"""
import math

import numpy as np
from enum import IntEnum
from typing import *
from colorama import init, Fore
from abc import abstractmethod, ABCMeta

init(autoreset=True)

__all__ = ["GoPlayer", "GoPoint", "GoString", "GoIllegalActionError", "GoBoardBase"]


class GoPlayer(IntEnum):
    black = 0
    none = 1
    white = 2
    b = 0
    n = 1
    w = 2

    @property
    def other(self):
        return GoPlayer(2 - self.value)

    @classmethod
    def to_player(cls, player: Optional[str]):
        try:
            return cls[player]
        except KeyError:
            return cls.none


class GoPoint(object):
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __add__(self, other: "GoPoint") -> "GoPoint":
        return GoPoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "GoPoint") -> "GoPoint":
        return GoPoint(self.x - other.x, self.y - other.y)

    def __mul__(self, other: "GoPoint") -> "GoPoint":
        return GoPoint(self.x * other.x, self.y * other.y)

    def __div__(self, other: "GoPoint") -> "GoPoint":
        return GoPoint(self.x // other.x, self.y // other.y)

    def __str__(self):
        return "(%s, %s)" % (self.x, self.y)

    def __abs__(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def __eq__(self, other: "GoPoint"):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return self.x | (self.y << 8)


class GoString(NamedTuple):
    player: GoPlayer
    stones: Set[GoPoint]
    liberties: Set[GoPoint]

    def is_dead(self) -> bool:
        return len(self.liberties) == 0

    def __len__(self) -> int:
        return len(self.stones)


class GoIllegalActionError(Exception):
    def __init__(self, action: Any, msg: str, board):
        super().__init__("%s: %s\n" % (msg, action))
        self.board = board

    def details(self):
        return self.board.details()

    @classmethod
    def move_out_of_range(cls, player: GoPlayer, point: GoPoint, board):
        return cls("%s %s" % (player.name, point), "move out of range", board)

    @classmethod
    def already_has_a_stone(cls, player: GoPlayer, point: GoPoint, board):
        return cls("%s %s" % (player.name, point), "already has a stone", board)

    @classmethod
    def commit_suicide(cls, player: GoPlayer, point: GoPoint, board):
        return cls("%s %s" % (player.name, point), "commit suicide", board)

    @classmethod
    def commit_robbery(cls, player: GoPlayer, point: GoPoint, board):
        return cls("%s %s" % (player.name, point), "commit robbery", board)

    @classmethod
    def illegal_player(cls, player: GoPlayer, board):
        return cls(player, "illegal player", board)


class GoBoardBase(metaclass=ABCMeta):

    def __init__(self, size=19):
        self._grid = np.full((size, size), GoPlayer.none, dtype=np.uint8)
        self._next_player = GoPlayer.black
        self.komi = 6.5

    def _setup_stones(self, player: GoPlayer, stones: Iterable[GoPoint] = None):
        if stones is not None:
            for stone in stones:
                self[stone] = player

    def setup_stones(self, black_stones: Optional[Iterable[GoPoint]] = None,
                     white_stones: Optional[Iterable[GoPoint]] = None,
                     empty_stones: Optional[Iterable[GoPoint]] = None):
        self._setup_stones(GoPlayer.black, black_stones)
        self._setup_stones(GoPlayer.white, white_stones)
        self._setup_stones(GoPlayer.none, empty_stones)

    def clean(self):
        self._grid = GoPlayer.none
        self._next_player = GoPlayer.black

    def setup_player(self, player: GoPlayer):
        self._next_player = player

    def get_neighbors(self, point: GoPoint, include_self=False) -> Tuple[GoPoint]:
        shape = self._grid.shape
        x, y = point
        if include_self:
            lst = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x, y))
        else:
            lst = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        return tuple(GoPoint(px, py) for px, py in lst if 0 <= px < shape[0] and 0 <= py < shape[1])

    def get_corners(self, point: GoPoint) -> Tuple[GoPoint]:
        shape = self._grid.shape
        x, y = point
        lst = ((x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1))
        return tuple(GoPoint(px, py) for px, py in lst if 0 <= px < shape[0] and 0 <= py < shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return self._grid.shape

    def neighbors(self, point: GoPoint) -> Iterator[GoPoint]:
        width, height = self._grid.shape
        pts = ((point.x - 1, point.y), (point.x + 1, point.y), (point.x, point.y - 1), (point.x, point.y + 1))
        return (GoPoint(x, y) for x, y in pts if 0 <= x < width and 0 <= y < height)

    def corners(self, point: GoPoint) -> Iterator[GoPoint]:
        width, height = self._grid.shape
        lst = ((point.x - 1, point.y - 1), (point.x - 1, point.y + 1), (point.x + 1, point.y - 1), (point.x + 1, point.y + 1))
        return (GoPoint(x, y) for x, y in lst if 0 <= x < width and 0 <= y < height)

    def score(self) -> float:
        black = 0
        white = self.komi
        for point in self:
            color = self._grid.item(point)
            if color == GoPlayer.black.value:
                black += 1
            elif color == GoPlayer.white.value:
                white += 1
            else:
                if all(self._grid.item(p) == GoPlayer.black.value for p in self.get_neighbors(point)):
                    black += 1
                elif all(self._grid.item(p) == GoPlayer.white.value for p in self.get_neighbors(point)):
                    white += 1
        return black - white

    def is_point_a_true_eye(self, point: GoPoint, player=GoPlayer.none) -> bool:
        if self[point] != GoPlayer.none:
            return False
        for neighbor in self.get_neighbors(point):
            if self[neighbor] != player:
                return False
        player_count = 0
        corners = self.get_corners(point)
        n_corners = len(corners)
        for corner in corners:
            if self[corner] == player:
                player_count += 1
        if n_corners == 4:
            return player_count >= 3
        elif n_corners == 2:
            return player_count == 2
        else: # n_corners == 1
            return player_count == 1

    @abstractmethod
    def get_string(self, point: GoPoint) -> Optional[GoString]:
        pass

    @abstractmethod
    def is_valid_point(self, point: GoPoint, color=GoPlayer.none) -> bool:
        pass

    @abstractmethod
    def play(self, point: Optional[GoPoint] = None, color=GoPlayer.none) -> Any:
        pass

    @property
    def next_player(self) -> GoPlayer:
        return self._next_player

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def __getitem__(self, point: GoPoint) -> GoPlayer:
        return GoPlayer(self._grid.item(tuple(point)))

    def __setitem__(self, point: GoPoint, player: GoPlayer):
        self._grid.itemset(tuple(point), player)

    def __iter__(self) -> Iterator[GoPoint]:
        X, Y = self._grid.shape
        for x in range(X):
            for y in range(Y):
                yield GoPoint(x, y)

    def __eq__(self, other):
        if isinstance(other, GoBoardBase):
            return self._next_player == other._next_player and \
                   self._grid.shape == other._grid.shape and \
                   np.all(self._grid == other._grid)
        return NotImplemented

    def details(self):
        X, Y = self._grid.shape
        rows = ["black:x white:o"]
        for x in range(X):
            cols = [""]
            for y in range(Y):
                color = self._grid.item((x, y))
                if color == GoPlayer.none.value:
                    cols.append(",")
                elif color == GoPlayer.white.value:
                    cols.append(Fore.WHITE + "o" + Fore.RESET)
                else:
                    cols.append(Fore.BLACK + 'x' + Fore.RESET)
            rows.append("".join(cols))
        rows.append("")
        return '\n'.join(rows)

    def __str__(self):
        return self.details()
