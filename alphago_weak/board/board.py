# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/7/7
@Version : 1.0
"""

from abc import *
from typing import *
import numpy as np
import collections
from .basic import *
from .zobrist_hash import zobrist_hash

__all__ = ["PyGoBoardBase", "GoBoard", "GoBoardAlpha", "GoBoardBeta"]


class PyGoBoardBase(GoBoardBase, metaclass=ABCMeta):

    def __init__(self, size=19):
        super().__init__(size)
        self._robbery: Optional[GoPoint] = None
        self.__hash = 0

    def place_stone(self, player: GoPlayer, stone: GoPoint):
        """Change a stone on the Go board with a revocable callable function.

        Args:
            stone: The position of the stone.
            player: The player who owns the target stone.
        Returns:
            A callable function which allows to withdraw the pre-action.
        """
        self.__hash ^= zobrist_hash(self[stone], stone) ^ zobrist_hash(player, stone)

    def __setitem__(self, pos: Optional[GoPoint], player: GoPlayer):
        if pos is not None:
            self.place_stone(player, pos)
            if player != GoPlayer.none:
                # assert self[pos] == GoPlayer.none
                dead_count = 0
                for neighbor in self.neighbors(pos):
                    string = self.get_string(neighbor)
                    if string is not None and string.player == player.other:
                        dead_count += len(string.stones)
                        self._robbery = next(iter(string.stones))
                        self.remove_string(string)
                if dead_count != 1:
                    self._robbery = None

    def __index__(self):
        return hash(self)

    def __int__(self):
        return hash(self)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return self.__hash

    def new_string(self, point: GoPoint):
        """Construct the chess string from the target point.

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

    def get_string(self, point: GoPoint) -> Optional[GoString]:
        """Search the chess string from a point.

        Args:
            point: Searching start point for the chess string.

        Returns:
            The string from the search if found, otherwise returns None.
        """
        return self.new_string(point)

    def get_strings(self) -> List[GoString]:
        """Search all strings from the board.

        Returns:
            A list of chess strings.
        """
        strings = []
        strings_map = np.full((self.size, self.size), False, dtype=np.bool_)
        for point in self:
            if not strings_map.item(tuple(point)):
                string = self.get_string(point)
                if string is not None:
                    for stone in string.stones:
                        strings_map.itemset(tuple(stone), True)
                    strings.append(string)
        return strings

    def remove_string(self, string: GoString):
        for stone in string.stones:
            self.place_stone(GoPlayer.none, stone)

    def is_valid_point(self, player, pos):
        if self[pos] != GoPlayer.none:
            return False
        neighbor_strings = []

        # has space for the stone
        for neighbor_string in map(self.get_string, self.neighbors(pos)):
            if neighbor_string is None:
                return True
            neighbor_strings.append(neighbor_string)

        if pos == self._robbery:
            return False

        for neighbor_string in neighbor_strings:
            if neighbor_string.player == player and len(neighbor_string.liberties) > 1:
                return True  # cannot commit suicide

        # any of this is true
        return any(map(lambda string: string == player.other and len(string) == 1, neighbor_strings))

    def valid_points(self, player):
        strings = self.get_strings()
        possibles = set(filter(lambda pos: self[pos] == GoPlayer.none, self))
        confuses = set()
        for string in strings:
            # assert len(string.liberties) >= 1
            if len(string.liberties) == 1:
                confuses.update(string.liberties)

        for confuse in confuses:
            if not self.is_valid_point(player, confuse):
                possibles.remove(confuse)

        def not_suicide(pos):
            return not all(map(lambda p: self[p] == player.other, self.neighbors(pos)))

        return filter(not_suicide, possibles)



class GoBoardAlpha(PyGoBoardBase):

    def __init__(self, size=19):
        super().__init__(size)
        self._grid = np.full((size, size), GoPlayer.none, dtype=np.uint8)

    def __getitem__(self, pos: GoPoint) -> GoPlayer:
        return GoPlayer(self._grid.item(tuple(pos)))

    def place_stone(self, player, stone):
        super().place_stone(player, stone)
        self._grid.itemset(tuple(stone),player)

    def score(self, player: GoPlayer, komi=6.5) -> float:
        counts = collections.Counter(self._grid.flat)
        if player == GoPlayer.black:
            komi = -komi
        return counts.get(player.value, 0) + komi - counts.get(player.other.value, 0)


class GoBoardBeta(PyGoBoardBase):
    def __init__(self, size=19):
        super().__init__(size)
        self._grid: Dict[GoPoint, GoString] = {}
        self.__robbery = None

    def __hash__(self):
        return self.__hash

    def __getitem__(self, point):
        string = self._grid.get(point, None)
        if string is None:
            return GoPlayer.none
        else:
            return string.player

    def __setitem__(self, point, player):
        if player == GoPlayer.none:
            pass

    def _remove_string(self, string: GoString):
        for stone in string.stones:
            del self._grid[stone]


GoBoard = GoBoardAlpha
