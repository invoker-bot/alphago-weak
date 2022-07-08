# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/11/19
@Version : 1.0
"""
import math
import collections
from abc import *
from typing import *
from enum import IntEnum
from collections import Counter

__all__ = ["GoPlayer", "GoEyeType", "GoPoint", "GoString", "GoIllegalActionError", "GoBoardBase"]


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
    def to_player(cls, player: Optional[str]) -> "GoPlayer":
        try:
            return cls[player]
        except KeyError:
            return cls.none

    def to_sgf(self) -> Optional[str]:
        if self == GoPlayer.white:
            return "w"
        elif self == GoPlayer.black:
            return "b"
        return None


class GoEyeType(IntEnum):
    none = 0
    fake = 1
    unknown = 2
    true = 3


class GoPoint(object):
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    @classmethod
    def to_point(cls, point: Optional[Tuple[int, int]]) -> Optional["GoPoint"]:
        if point is None:
            return None
        else:
            return cls(*point)

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __add__(self, other: "GoPoint"):
        return GoPoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "GoPoint"):
        return GoPoint(self.x - other.x, self.y - other.y)

    def __mul__(self, other: "GoPoint"):
        return GoPoint(self.x * other.x, self.y * other.y)

    def __div__(self, other: "GoPoint"):
        return GoPoint(self.x // other.x, self.y // other.y)

    def __str__(self):
        return "(%d, %d)" % (self.x, self.y)

    def __repr__(self):
        return "GoPoint(%d, %d)" % (self.x, self.y)

    def __abs__(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def __eq__(self, other: "GoPoint"):
        return hash(self) == hash(other)

    def __hash__(self):
        return self.x | (self.y << 8)


class GoString(NamedTuple):
    player: GoPlayer
    stones: Set[GoPoint]
    liberties: Set[GoPoint] = set()

    def is_dead(self) -> bool:
        return len(self.liberties) == 0

    def __len__(self) -> int:
        return len(self.stones)

    def __eq__(self, other: "GoString"):
        return self.player == other.player and self.stones == other.stones

    def __hash__(self):
        return (len(self.liberties) << 16) | (len(self.stones) << 8) | self.player.value


class GoIllegalActionError(Exception):
    def __init__(self, player: GoPlayer, pos: GoPoint):
        super().__init__("%s - %s\n" % (player, pos))


class GoBoardBase(metaclass=ABCMeta):

    def __init__(self, size=19):
        """Initialize a go board with specified shape with (size, size).

        Args:
            size:
        """
        if not 0 < size <= 25:
            raise ValueError("board size must be within [1, 25], actual: %d" % size)
        self._size = size

    @property
    def size(self):
        return self._size

    def _setup_stones(self, player: GoPlayer, stones: Iterable[GoPoint]):
        """Change stones on the Go board without checking.

        Notes:
            * Improper application may lead to an invalid situation.
        Args:
            player: The player who owns the target stones.
            stones: The positions of the stones.
        """
        for stone in stones:
            self[stone] = player

    def setup_stones(self, black_stones: Iterable[GoPoint], white_stones: Iterable[GoPoint], empty_stones: Iterable[GoPoint]):
        self._setup_stones(GoPlayer.black, black_stones)
        self._setup_stones(GoPlayer.white, white_stones)
        self._setup_stones(GoPlayer.none, empty_stones)

    def neighbors(self, point: GoPoint) -> Iterable[GoPoint]:
        pts = ((point.x - 1, point.y), (point.x + 1, point.y), (point.x, point.y - 1), (point.x, point.y + 1))
        return (GoPoint(x, y) for x, y in pts if 0 <= x < self.size and 0 <= y < self.size)

    def corners(self, point: GoPoint) -> Iterable[GoPoint]:
        lst = ((point.x - 1, point.y - 1), (point.x - 1, point.y + 1), (point.x + 1, point.y - 1), (point.x + 1, point.y + 1))
        return (GoPoint(x, y) for x, y in lst if 0 <= x < self.size and 0 <= y < self.size)

    def is_eye_point(self, player: GoPlayer, point: GoPoint) -> bool:
        assert player != GoPlayer.none
        if self[point] != GoPlayer.none:
            return False
        return all(map(lambda p: self[p] == player, self.neighbors(point)))

    def eye_type(self, player: GoPlayer, point: GoPoint) -> GoEyeType:
        if not self.is_eye_point(player, point):
            return GoEyeType.none

        corners = tuple(map(lambda p: self[p], self.corners(point)))
        counter = Counter(corners)
        n_corners = len(corners)

        if n_corners == 4:
            if counter[player] >= 3:
                return GoEyeType.true
            elif counter[player.other] > 1:
                return GoEyeType.fake
            return GoEyeType.unknown
        else:  # n_corners == 2 or 1
            if counter[player] == n_corners:
                return GoEyeType.true
            elif counter[player.other] > 0:
                return GoEyeType.fake
            return GoEyeType.unknown

    def __iter__(self) -> Iterator[GoPoint]:
        for x in range(self.size):
            for y in range(self.size):
                yield GoPoint(x, y)

    def __len__(self) -> int:
        return self.size * self.size

    def items(self) -> Iterator[Tuple[GoPoint, GoPlayer]]:
        for pos in self:
            yield pos, self[pos]

    @abstractmethod
    def __getitem__(self, pos: GoPoint) -> GoPlayer:
        ...

    @abstractmethod
    def __setitem__(self, pos: GoPoint, player: Union[GoPlayer, int]):
        """Change a stone on the Go board without checking.
        Notes:
            * Improper usage may violate the rule of the Game of Go.
        Args:
            pos: The position of the stone.
            player: The player who owns the target stone placed.
        """

    def __delitem__(self, point: GoPoint):
        self.__setitem__(point, GoPlayer.none)

    def play(self, player: GoPlayer, pos: Optional[GoPoint] = None):
        """Take a single step with exception security.

        Args:
            player: The target player.
            pos: The target point.
        Returns:
            A callable function which allows to withdraw the pre-action.
        Raises:
            GoIllegalActionError: When performing an illegal action.
        """
        if pos is not None:
            if not self.is_valid_point(player, pos):
                raise GoIllegalActionError(player, pos)
            self[pos] = player

    @abstractmethod
    def is_valid_point(self, player: GoPlayer, pos: GoPoint) -> bool:
        """Judge whether a point can be placed by a player with no side effects.

        Args:
            player: The target player.
            pos: The target point.
        Returns:
            Whether the placement action is valid.
        """
        ...

    def valid_points(self, player: GoPlayer) -> Iterable[GoPoint]:
        return (pos for pos in self if self.is_valid_point(player, pos))

    def score(self, player: GoPlayer, komi=6.5) -> float:
        counts = collections.Counter(map(self.__getitem__, iter(self)))
        if player == GoPlayer.black:
            komi = -komi
        return counts.get(player.value, 0) + komi - counts.get(player.other.value, 0)
