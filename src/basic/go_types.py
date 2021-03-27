# -*- coding: utf-8 -*-

import numpy as np
from enum import IntEnum
from typing import *
from colorama import init, Fore
from abc import abstractmethod, ABCMeta
from itertools import product

init(autoreset=True)

__all__ = ["GoPlayer", "GoPoint", "GoString", "GoIllegalActionError",
           "GoBoardBase"]


class GoPlayer(IntEnum):
    black = 0
    white = 1
    none = 2

    @property
    def other(self):
        if self == GoPlayer.none:
            return self
        return GoPlayer.white if self == GoPlayer.black else GoPlayer.black

    @staticmethod
    def to_player(value):
        return {"w": GoPlayer.white, "white": GoPlayer.white, GoPlayer.white: GoPlayer.white, 1: GoPlayer.white,
                "b": GoPlayer.black, "black": GoPlayer.black, GoPlayer.black: GoPlayer.black, 0: GoPlayer.black,
                "None": GoPlayer.none, "none": GoPlayer.none, None: GoPlayer.none}[value]


GoPoint = Tuple[int, int]


class GoString(NamedTuple):
    player: GoPlayer
    stones: Set[GoPoint]
    liberties: Set[GoPoint]

    def is_dead(self):
        return len(self.liberties) == 0

    def __len__(self):
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

    def __init__(self, shape: int = 19, komi=6.5, first_player=GoPlayer.black):
        self._next_player = first_player
        self._grid = np.full((shape, shape), GoPlayer.none.value, dtype=np.uint8)
        self.komi = komi

    def setup_stones(self, black_stones: Optional[Iterable[GoPoint]] = None,
                     white_stones: Optional[Iterable[GoPoint]] = None,
                     empty_stones: Optional[Iterable[GoPoint]] = None) -> NoReturn:
        if black_stones:
            for stone in black_stones:
                self._grid.itemset(stone, GoPlayer.black.value)
        if white_stones:
            for stone in white_stones:
                self._grid.itemset(stone, GoPlayer.white.value)

    def setup_player(self, player: GoPlayer) -> NoReturn:
        self._next_player = player

    def get_neighbors(self, point: GoPoint, include_self=False) -> Tuple[GoPoint]:
        shape = self._grid.shape
        x, y = point
        if include_self:
            lst = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x, y))
        else:
            lst = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        return tuple((px, py) for px, py in lst if 0 <= px < shape[0] and 0 <= py < shape[1])

    def get_corners(self, point: GoPoint) -> Tuple[GoPoint]:
        shape = self._grid.shape
        x, y = point
        lst = ((x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1))
        return tuple((px, py) for px, py in lst if 0 <= px < shape[0] and 0 <= py < shape[1])

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

    def is_point_a_true_eye(self, point: GoPoint, color=GoPlayer.none) -> bool:
        player = self._next_player if color == GoPlayer.none else color
        for neighbor in self.get_neighbors(point):
            if self._grid.item(neighbor) != player.value:
                return False
        other_count = 0
        player_count = 0
        for corner in self.get_corners(point):
            if self._grid.item(corner) == player.value:
                player_count += 1
            else:
                other_count += 1
        if other_count == 0 or (other_count == 1 and player_count == 3):
            return True
        return False

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
        return GoPlayer(self._grid.item(point))

    def __iter__(self) -> Iterator[GoPoint]:
        X, Y = self._grid.shape
        yield from product(range(X), range(Y))

    def __eq__(self, other):
        if isinstance(other, GoBoardBase):
            return self._next_player == other._next_player and \
                   self._grid.shape == other._grid.shape and \
                   np.all(self._grid == other._grid)
        return NotImplemented

    def summary(self):
        x, y = self._grid.shape

        return "\tshape:(%d,%d)" % (x, y)

    def details(self):
        shape = self._grid.shape
        rows = ["black:x white:o"]
        for x in range(shape[0]):
            cols = [""]
            for y in range(shape[1]):
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
