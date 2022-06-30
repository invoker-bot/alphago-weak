# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/11/19
@Version : 1.0
"""

import numpy as np
from typing import *
from .basic import *
from .zobrist_hash import zobrist_hash

__all__ = ["GoBoardBase", "GoBoard"]


class GoBoard(GoBoardBase):

    def __init__(self, size=19):
        super().__init__(size)
        self.__hash = 0
        self.__robbery = None

    def clean(self):
        super().clean()
        self.__hash = 0
        self.__robbery = None

    def __hash__(self):
        return self.__hash

    def __index__(self):
        return hash(self)

    def __eq__(self, other):
        if isinstance(other, GoBoard):
            return hash(self) == hash(other)
        return NotImplemented

    def __int__(self):
        return hash(self)

    def __delitem__(self, point: GoPoint):
        self.__setitem__(point, GoPlayer.none)

    def __len__(self) -> int:
        width, height = self._grid.shape
        return width * height

    def _place_stone(self, pos: GoPoint, player: Union[GoPlayer, int] = GoPlayer.none):
        """Change a stone on the Go board without checking.

        Notes:
            * Improper application may lead to an invalid situation.
        Args:
            pos: The position of the stone.
            player: The player who owns the target stone.
        """
        self.__hash ^= zobrist_hash(self[pos], pos) ^ zobrist_hash(player.value, pos)
        self[pos] = player

    def _place_stones(self, stones: Iterable[GoPoint], player: Union[GoPlayer, int] = GoPlayer.none):
        """Change stones on the Go board without checking.

        Notes:
            * Improper application may lead to an invalid situation.
        Args:
            stones: The positions of the stones.
            player: The player who owns the target stones.
        """
        for stone in stones:
            self._place_stone(stone, player)

    def place_stone(self, stone: GoPoint, player: Union[GoPlayer, int] = GoPlayer.none) -> Callable[[], None]:
        """Change a stone on the Go board with a revocable callable function.

        Args:
            stone: The position of the stone.
            player: The player who owns the target stone.
        Returns:
            A callable function which allows to withdraw the pre-action.
        """
        former_stone: int = self[stone]

        def back():
            self._place_stone(stone, former_stone)

        self._place_stone(stone, player)
        return back

    def place_stones(self, stones: Iterable[GoPoint], player: Union[GoPlayer, int] = GoPlayer.none) -> Callable[[], None]:
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
                self._place_stone(stone, player)

        self._place_stones(stones, player)
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

    def is_valid_point(self, point: GoPoint, player: Union[GoPlayer, int] = GoPlayer.none) -> bool:
        """Judge whether a point can be placed by a player with no side effects.

        Args:
            point: The target point.
            player: The target player.
        Returns:
            Whether the placement action is valid.
        """
        try:
            back = self.play(point, player)
            back()
            return True
        except GoIllegalActionError:
            return False

    def valid_points(self, player: Union[GoPlayer, int] = GoPlayer.none) -> np.ndarray:
        strings = self.get_strings()
        tensor = np.zeros(self._grid.shape)
        for pos in self:
            if self[pos] == GoPlayer.none:
                tensor.itemset(pos, 1)
        confuses = set()
        for string in strings:
            if len(string.liberties) == 1:
                confuses.update(string.liberties)
        for confuse in confuses:
            if self.is_valid_point(confuse):
                tensor.itemset(confuse, 1)
            else:
                tensor.itemset(confuse, 0)
        return tensor

    def is_eye_point(self, point: GoPoint, player: Union[GoPlayer, int] = GoPlayer.none) -> bool:
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

    def is_true_eye_point(self, point: GoPoint, player: Union[GoPlayer, int] = GoPlayer.none) -> bool:
        if not self.is_eye_point(point, player):
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

    def neighbor_dead_stones(self, point: GoPoint, player: Union[GoPlayer, int]) -> Set[GoPoint]:
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

    def play(self, point: Optional[GoPoint] = None, player: GoPlayer = GoPlayer.none) -> Callable[[], None]:
        """Take a single step with exception security.

        Args:
            point: The target point.
            player: The target player.
        Returns:
            A callable function which allows to withdraw the pre-action.
        Raises:
            GoIllegalActionError: When performing an illegal action.
        """
        if point is not None:
            try:
                if self[point] == GoPlayer.none:
                    back1 = self.place_stone(point, player)
                    dead_stones = self.neighbor_dead_stones(point, player.other)
                    robbery = None
                    if len(dead_stones) == 1:
                        robbery = (next(iter(dead_stones)), point)
                        if self.__robbery is not None and robbery == tuple(reversed(self.__robbery)):
                            back1()
                            raise GoIllegalActionError.commit_robbery(player, point, self)

                    tmp_robbery = self.__robbery
                    self.__robbery = robbery
                    back2 = self.place_stones(dead_stones, GoPlayer.none)

                    def back():
                        back2()
                        self.__robbery = tmp_robbery
                        back1()

                    if self.get_string(point).is_dead():
                        back()
                        raise GoIllegalActionError.commit_suicide(player, point, self)
                    return back
                else:  # self[point] != GoPlayer.none
                    raise GoIllegalActionError.already_has_a_stone(player, point, self)
            except IndexError:
                raise GoIllegalActionError.move_out_of_range(player, point, self)
