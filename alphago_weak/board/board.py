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
from .basic import *
from .zobrist_hash import zobrist_hash

__all__ = ["PyGoBoardBase", "GoBoard", "GoBoardAlpha", "GoBoardBeta"]


class PyGoBoardBase(GoBoardBase, metaclass=ABCMeta):

    def __init__(self, size=19):
        super().__init__(size)
        self._robbery: Optional[GoPoint] = None
        self.__hash = 0

    def _place_stone(self, player: GoPlayer, stone: GoPoint):
        """Change a stone on the Go board with a revocable callable function.

        Args:
            stone: The position of the stone.
            player: The player who owns the target stone.
        Returns:
            A callable function which allows to withdraw the pre-action.
        """
        self.__hash ^= zobrist_hash(self[stone], stone) ^ zobrist_hash(player, stone)

    def _remove_stone(self, stone: GoPoint):
        self._place_stone(GoPlayer.none, stone)

    def __setitem__(self, pos, player):
        if pos is None:
            self._robbery = None
        else:
            if player != GoPlayer.none:
                # assert self[pos] == GoPlayer.none
                dead_count = 0
                for neighbor in self.neighbors(pos):
                    string = self.get_string(neighbor)
                    if string is not None and string.player == player.other and len(string.liberties) == 1:
                        dead_count += len(string.stones)
                        self._robbery = next(iter(string.stones))
                        self.remove_string(string)
                if dead_count != 1:
                    self._robbery = None
                self._place_stone(player, pos)
            else:
                self._remove_stone(pos)

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

    def get_strings(self) -> Iterable[GoString]:
        """Search all strings from the board.

        Returns:
            A list of chess strings.
        """
        # strings = []
        strings_map = np.full((self.size, self.size), False, dtype=np.bool_)
        for point in self:
            if not strings_map.item(tuple(point)):
                string = self.get_string(point)
                if string is not None:
                    for stone in string.stones:
                        strings_map.itemset(tuple(stone), True)
                    yield string

    def remove_string(self, string: GoString):
        for stone in string.stones:
            self._place_stone(GoPlayer.none, stone)

    def is_valid_point(self, player, pos):
        if self[pos] != GoPlayer.none:
            return False

        neighbor_strings: Set[GoString] = set()

        # has space for the stone
        for neighbor_string in map(self.get_string, self.neighbors(pos)):
            if neighbor_string is None:
                return True
            neighbor_strings.add(neighbor_string)

        for neighbor_string in neighbor_strings:
            if neighbor_string.player == player and len(neighbor_string.liberties) > 1:
                return True  # not commit suicide

        dead_count = 0
        for neighbor_string in neighbor_strings:
            if neighbor_string.player == player.other and len(neighbor_string.liberties) == 1:
                dead_count += len(neighbor_string.stones)
        if dead_count == 1:
            return pos != self._robbery
        else:
            return dead_count > 0

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

    def encode(self, arr, offset, encode_type, player, length=1):
        if encode_type == GoBoardEncodeType.player_stone_liberties:
            for string in self.get_strings():
                if string.player == player:
                    liberties_offset = offset + min(len(string.liberties), length) - 1
                    for pos in string.stones:
                        arr.itemset((liberties_offset, pos.x, pos.y), 1)
        else:
            super().encode(arr, offset, encode_type, player, length=length)

class GoBoardAlpha(PyGoBoardBase):

    def __init__(self, size=19):
        super().__init__(size)
        self._grid = np.full((size, size), GoPlayer.none, dtype=np.uint8)

    def __getitem__(self, pos: GoPoint) -> GoPlayer:
        return GoPlayer(self._grid.item(tuple(pos)))

    def _place_stone(self, player, stone):
        super()._place_stone(player, stone)
        self._grid.itemset(tuple(stone),player)


class GoBoardBeta(PyGoBoardBase):
    def __init__(self, size=19):
        super().__init__(size)
        self._grid: Dict[GoPoint, GoString] = {}

    def __getitem__(self, point):
        string = self._grid.get(point, None)
        if string is None:
            return GoPlayer.none
        else:
            return string.player

    def _place_stone(self, player, stone):
        super()._place_stone(player, stone)
        if player == GoPlayer.none:
            self._grid.pop(stone, None)
        else:
            neighbor_strings = list(filter(lambda string: string is not None and string.player==player, map(self.get_string, self.neighbors(stone))))
            if len(neighbor_strings) == 0:
                self._grid[stone] = GoString(player, {stone}, set(self.neighbors(stone)))
            else:
                string = max(neighbor_strings, key=lambda string: len(string.stones))
                self._grid[stone] = string
                for other_string in neighbor_strings:
                    if other_string is not string:
                        for other_stone in other_string.stones:
                            self._grid[other_stone] = string
                        string.stones.update(other_string.stones)
                        string.liberties.update(other_string.liberties)
                string.stones.add(stone)
                string.liberties.remove(stone)
                for neighbor in self.neighbors(stone):
                    if self.get_string(neighbor) is None:
                        string.liberties.add(neighbor)


GoBoard = GoBoardBeta
