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
        self.__hash = 0

    def place_stone(self, player: GoPlayer, stone: GoPoint):
        self.__hash ^= zobrist_hash(self[stone], stone) ^ zobrist_hash(player, stone)

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

    def is_valid_point(self, player, pos):
        if self[pos] != GoPlayer.none:
            return False
        neighbor_strings = [self.get_string(neighbor) for neighbor in self.neighbors(pos)]
        # has space for the stone
        for neighbor_string in neighbor_strings:
            if neighbor_string is None:
                return True

        for neighbor_string in neighbor_strings:
            if neighbor_string.player == player and len(neighbor_string.liberties) > 1:
                return True  # cannot commit suicide

        # any of this is true
        return any(map(lambda string: string == player.other and len(string) == 1, neighbor_strings))


class GoBoardAlpha(PyGoBoardBase):

    def __init__(self, size=19):
        super().__init__(size)
        self._grid = np.full((size, size), GoPlayer.none, dtype=np.uint8)
        self.__robbery = None

    def __getitem__(self, pos: GoPoint) -> GoPlayer:
        return GoPlayer(self._grid.item(tuple(pos)))

    def __setitem__(self, pos: GoPoint, player: GoPlayer):
        self._grid.itemset(tuple(pos), player)
        super().__setitem__(pos, player)

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
        former_stone = self[stone]

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

    def is_valid_point(self, player: GoPlayer, pos: GoPoint):
        try:
            back = self.play(player, pos)
            back()
            return True
        except GoIllegalActionError:
            return False

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

    def score(self, player: GoPlayer, komi=6.5) -> float:
        counts = collections.Counter(self._grid.flat)
        if player == GoPlayer.black:
            komi = -komi
        return counts.get(player.value, 0) + komi - counts.get(player.other.value, 0)

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
            if self[pos] == GoPlayer.none:
                back1 = self.place_stone(player, pos)
                dead_stones = self.neighbor_dead_stones(player.other, pos)
                robbery = None
                if len(dead_stones) == 1:
                    robbery = (next(iter(dead_stones)), pos)
                    if self.__robbery is not None and robbery == tuple(reversed(self.__robbery)):
                        back1()
                        raise GoIllegalActionError(player, pos)
                tmp_robbery = self.__robbery
                self.__robbery = robbery
                back2 = self.place_stones(GoPlayer.none, dead_stones)

                def back():
                    back2()
                    self.__robbery = tmp_robbery
                    back1()

                if self.get_string(pos).is_dead():
                    back()
                    raise GoIllegalActionError(player, pos)
                return back
            else:  # self[point] != GoPlayer.none
                raise GoIllegalActionError(player, pos)


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

    def _remove_string(self, string: GoString):
        for stone in string.stones:
            del self._grid[stone]





GoBoard = GoBoardAlpha
