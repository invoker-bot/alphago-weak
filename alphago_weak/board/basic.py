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
from abc import abstractmethod, ABCMeta

from .zobrist_hash import zobrist_hash

__all__ = ["GoPlayer", "GoPoint", "GoString", "GoIllegalActionError", "GoBoardProtocol", "GoBoardBase", "GoBoardAlpha"]


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

    def __eq__(self, other: "GoString"):
        return self.player == other.player and self.stones == other.stones

    def __hash__(self):
        return (len(self.liberties) << 16) | (len(self.stones) << 8) | self.player.value


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


class GoBoardProtocol(Protocol):

    def __init__(self, size=19, komi=6.5):
        self._size = size
        self.komi = komi

    @property
    def size(self) -> int:
        return self._size

    def clean(self):
        ...

    def __iter__(self) -> Iterator[GoPoint]:
        ...

    def __getitem__(self, pos: GoPoint) -> GoPlayer:
        ...

    def __setitem__(self, pos: GoPoint, player: GoPlayer):
        """Change a stone on the Go board without checking.

        Notes:
            * Improper usage may violate the rule of the Game of Go.
        Args:
            pos: The position of the stone.
            player: The player who owns the target stone placed.
        """
        ...

    def play(self, player, pos: Optional[GoPoint] = None) -> Any:
        """Take a single step with exception security.

        Args:
            player: The target player.
            pos: The target point.
        Returns:
            A callable function which allows to withdraw the pre-action.
        Raises:
            GoIllegalActionError: When performing an illegal action.
        """
        ...

    def is_valid_point(self, player: GoPlayer, pos: GoPoint) -> bool:
        """Judge whether a point can be placed by a player with no side effects.

        Args:
            player: The target player.
            pos: The target point.
        Returns:
            Whether the placement action is valid.
        """
        ...

    def get_string(self, point: GoPoint) -> Optional[GoString]:
        ...

    def get_strings(self) -> List[GoString]:
        ...


class GoBoardBase(GoBoardProtocol, metaclass=ABCMeta):

    def __init__(self, size=19, komi=6.5):
        if not 0 < size <= 25:
            raise ValueError("board size must be within [1, 25], actual: %d" % size)
        super().__init__(size, komi)
        self._size = size
        self.komi = komi

    @abstractmethod
    def clean(self):
        ...

    def _setup_stones(self, player: Union[GoPlayer, int], stones: Iterable[GoPoint]):
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

    def get_neighbors(self, point: GoPoint, include_self=False) -> Tuple[GoPoint]:
        x, y = point
        if include_self:
            lst = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x, y))
        else:
            lst = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        return tuple(GoPoint(px, py) for px, py in lst if 0 <= px < self.size and 0 <= py < self.size)

    def get_corners(self, point: GoPoint) -> Tuple[GoPoint]:
        x, y = point
        lst = ((x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1))
        return tuple(GoPoint(px, py) for px, py in lst if 0 <= px < self.size and 0 <= py < self.size)

    def neighbors(self, point: GoPoint) -> Iterator[GoPoint]:
        pts = ((point.x - 1, point.y), (point.x + 1, point.y), (point.x, point.y - 1), (point.x, point.y + 1))
        return (GoPoint(x, y) for x, y in pts if 0 <= x < self.size and 0 <= y < self.size)

    def corners(self, point: GoPoint) -> Iterator[GoPoint]:
        lst = ((point.x - 1, point.y - 1), (point.x - 1, point.y + 1), (point.x + 1, point.y - 1), (point.x + 1, point.y + 1))
        return (GoPoint(x, y) for x, y in lst if 0 <= x < self.size and 0 <= y < self.size)

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
        else:  # n_corners == 1
            return player_count == 1

    def valid_points(self, player: Union[GoPlayer, int]) -> Set[GoPoint]:
        return set(pos for pos in self if self.is_valid_point(player, pos))

    @abstractmethod
    def play(self, player, point: Optional[GoPoint] = None) -> Any:
        ...

    def __iter__(self) -> Iterator[GoPoint]:
        for x in range(self.size):
            for y in range(self.size):
                yield GoPoint(x, y)

    def __len__(self) -> int:
        return self.size * self.size

    def __index__(self):
        return hash(self)

    def __int__(self):
        return hash(self)

    def __eq__(self, other):
        return hash(self) == hash(other)


class GoBoardAlpha(GoBoardBase):

    def __init__(self, size=19, komi=6.5):
        super().__init__(size, komi)
        self._grid = np.full((size, size), GoPlayer.none, dtype=np.uint8)
        self.__hash = 0
        self.__robbery = None

    def clean(self):
        self._grid[:] = GoPlayer.none
        self.__hash = 0
        self.__robbery = None

    def __hash__(self):
        return self.__hash

    def __getitem__(self, pos: GoPoint) -> GoPlayer:
        return GoPlayer(self._grid.item(tuple(pos)))

    def __setitem__(self, pos: GoPoint, player: Union[GoPlayer, int]):
        self._grid.itemset(tuple(pos), player)
        self.__hash ^= zobrist_hash(self[pos], pos) ^ zobrist_hash(player, pos)

    def __delitem__(self, point: GoPoint):
        self.__setitem__(point, GoPlayer.none)

    def place_stone(self, player: GoPlayer, stone: GoPoint) -> Callable[[], None]:
        """Change a stone on the Go board with a revocable callable function.

        Args:
            stone: The position of the stone.
            player: The player who owns the target stone.
        Returns:
            A callable function which allows to withdraw the pre-action.
        """
        former_stone: int = self[stone]

        def back():
            self[stone] = former_stone

        self[stone] = player
        return back

    def place_stones(self, player: GoPlayer, stones: Iterable[GoPoint]) -> Callable[[], None]:
        """Change stones on the Go board with a revocable callable function.

        Args:
            stones: The positions of the stones.
            player: The player who owns the target stones.

        Returns:
            A callable function which allows to withdraw the pre-action.
        """
        former_stones = {stone: self[stone] for stone in stones}

        def back():
            for stone, player in former_stones.items():
                self[stone] = player

        self._setup_stones(player, stones)
        return back

    def get_string(self, point: GoPoint) -> Optional[GoString]:
        """Search the chess string from a point.

        Args:
            point: Searching start point for the chess string.

        Returns:
            The string from the search if found, otherwise returns None.
        """
        player = self[point]
        if player != GoPlayer.none:
            string = GoString(GoPlayer(player), {point}, set())
            neighbors_queue = list(self.neighbors(point))
            while len(neighbors_queue) != 0:
                point = neighbors_queue.pop()
                point_t = self[point]
                if point_t == GoPlayer.none:  # liberty
                    string.liberties.add(point)
                elif point_t == player and point not in string.stones:  # stone
                    string.stones.add(point)
                    neighbors_queue.extend(self.neighbors(point))
            return string

    def get_strings(self) -> List[GoString]:
        """Search all strings from the board.

        Returns:
            A list of chess strings.
        """
        strings = []
        strings_map = np.full(self._grid.shape, False, dtype=np.bool_)
        for point in self:
            if not strings_map.item(tuple(point)):
                string = self.get_string(point)
                if string is not None:
                    for stone in string.stones:
                        strings_map.itemset(tuple(stone), True)
                    strings.append(string)
        return strings

    def is_valid_point(self, player: GoPlayer, pos: GoPoint):
        try:
            back = self.play(player, pos)
            back()
            return True
        except GoIllegalActionError:
            return False

    def valid_points(self, player: GoPlayer) -> Set[GoPoint]:
        strings = self.get_strings()
        possible = set(filter(lambda pos: self[pos] == GoPlayer.none, self))
        confuses = set()
        for string in strings:
            if len(string.liberties) == 1:
                confuses.update(string.liberties)
        for confuse in confuses:
            if not self.is_valid_point(player, confuse):
                possible.remove(confuse)

        def not_suicide(pos):
            return not all(map(lambda p: self[p] == player.other, self.get_neighbors(pos)))

        return set(filter(not_suicide, possible))

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

    def is_eye_point(self, player: Union[GoPlayer, int], point: GoPoint) -> bool:
        if self[point] != GoPlayer.none:
            return False
        neighbors = self.neighbors(point)
        if player == GoPlayer.none:
            neighbor_t = self[next(neighbors)]
            if neighbor_t == player.none:
                return False
            for _neighbor in neighbors:
                if neighbor_t != self[_neighbor]:
                    return False
        else:
            for neighbor in neighbors:
                if player != neighbor:
                    return False
        return True

    def is_true_eye_point(self, player: GoPlayer, point: GoPoint) -> bool:
        if not self.is_eye_point(player, point):
            return False
        corners = list(self.corners(point))
        if player == GoPlayer.none:
            player = self[next(self.neighbors(point))]
        count = 0
        for corner in corners:
            if self[corner] == player:
                count += 1
        if len(corners) == 4:
            return count >= 3
        else:
            return len(corners) == count

    def neighbor_dead_stones(self, player: Union[GoPlayer, int], point: GoPoint) -> Set[GoPoint]:
        """Get the chess stones of the player died near the point.

        Args:
            point: The target point.
            player: The target player.

        Returns:
            The dead stones owned by the player.
        """
        dead_points = set()
        for point in self.neighbors(point):
            if self[point] == player:
                string = self.get_string(point)
                if string.is_dead():
                    dead_points.update(string.stones)
        return dead_points

    def play(self, player: GoPlayer, pos: Optional[GoPoint] = None) -> Callable[[], None]:
        if pos is not None:
            try:
                if self[pos] == GoPlayer.none:
                    back1 = self.place_stone(player, pos)
                    dead_stones = self.neighbor_dead_stones(player.other, pos)
                    robbery = None
                    if len(dead_stones) == 1:
                        robbery = (next(iter(dead_stones)), pos)
                        if self.__robbery is not None and robbery == tuple(reversed(self.__robbery)):
                            back1()
                            raise GoIllegalActionError.commit_robbery(player, pos, self)

                    tmp_robbery = self.__robbery
                    self.__robbery = robbery
                    back2 = self.place_stones(GoPlayer.none, dead_stones)

                    def back():
                        back2()
                        self.__robbery = tmp_robbery
                        back1()

                    if self.get_string(pos).is_dead():
                        back()
                        raise GoIllegalActionError.commit_suicide(player, pos, self)
                    return back
                else:  # self[point] != GoPlayer.none
                    raise GoIllegalActionError.already_has_a_stone(player, pos, self)
            except IndexError:
                raise GoIllegalActionError.move_out_of_range(player, pos, self)
