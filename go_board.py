# -*- coding: utf-8 -*-

import collections
from typing import *
import numpy as np
import zobrist_hash
from go_types import *

__all__ = ["GoBoard"]


class GoBoard(GoBoardBase):

    def __init__(self, shape: int = 19, first_player=GoPlayer.black):
        super().__init__(shape, first_player)
        self._hash = 0
        self.__robbery = None

    def __hash__(self):
        return self._hash

    def __index__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, GoBoard):
            return self._hash == other._hash
        return NotImplemented

    def __int__(self):
        return self._hash

    def _place_stone(self, stone: GoPoint, player: GoPlayer = GoPlayer.none):
        self._hash ^= zobrist_hash.get_hash(self._grid.item(stone), stone) ^ zobrist_hash.get_hash(player.value, stone)
        self._grid.itemset(stone, player.value)

    def _place_stones(self, stones: Iterable[GoPoint], player: GoPlayer = GoPlayer.none):
        for stone in stones:
            self._place_stone(stone, player)

    def place_stones(self, stones: Iterable[GoPoint], player: GoPlayer = GoPlayer.none):
        former_stones = {stone: self._grid.item(stone) for stone in stones}

        def back():
            for stone, color in former_stones.items():
                self._place_stone(stone, GoPlayer(color))

        self._place_stones(stones, player)
        return back

    def get_string(self, point: GoPoint) -> Optional[GoString]:
        color = self._grid.item(point)
        if color == GoPlayer.none.value:
            return None
        else:
            string = GoString(GoPlayer(color), {point}, set())
            neighbors_queue = list(self.get_neighbors(point))
            while len(neighbors_queue) != 0:
                point = neighbors_queue.pop()
                point_t = self._grid.item(point)
                if point_t == GoPlayer.none.value:  # liberty
                    string.liberties.add(point)
                elif point_t == color and point not in string.stones:  # stone
                    string.stones.add(point)
                    neighbors_queue.extend(self.get_neighbors(point))
            return string

    def get_strings(self) -> List[GoString]:
        strings = []
        strings_map = np.full(self._grid.shape, False, dtype=np.bool_)
        for point in self:
            if not strings_map.item(point):
                string = self.get_string(point)
                if string is not None:
                    for stone in string.stones:
                        strings_map.itemset(stone, True)
                    strings.append(string)
        return strings

    def is_valid_point(self, point: GoPoint, color=GoPlayer.none) -> bool:
        try:
            back = self.play(point, color)
            back()
            return True
        except GoIllegalActionError:
            return False

    def get_dead_stones(self, point: GoPoint, player: GoPlayer) -> Set[GoPoint]:
        dead_points = set()
        for point in self.get_neighbors(point, True):
            if self._grid.item(point) == player.value:
                string = self.get_string(point)
                if string is not None and string.is_dead():
                    dead_points.update(string.stones)
        return dead_points

    def play(self, point: Optional[GoPoint] = None, color=GoPlayer.none) -> Callable[[], NoReturn]:
        if color != GoPlayer.none:
            self._next_player = color
        player = self._next_player
        if point is None:
            self._next_player = player.other

            def back():
                self._next_player = player

            return back
        else:
            try:
                if self._grid.item(point) == GoPlayer.none.value:
                    back1 = self.place_stones((point,), self._next_player)
                    dead_stones = self.get_dead_stones(point, self._next_player.other)
                    robbery = None
                    if len(dead_stones) == 1:
                        robbery = (next(iter(dead_stones)), point)
                        if self.__robbery and robbery == (self.__robbery[1], self.__robbery[0]):
                            back1()
                            raise GoIllegalActionError.commit_robbery(self._next_player, point, self)

                    tmp_robbery = self.__robbery
                    self.__robbery = robbery
                    back2 = self.place_stones(dead_stones, GoPlayer.none)

                    def back():
                        back2()
                        back1()
                        self.__robbery = tmp_robbery
                        self._next_player = player

                    if len(self.get_dead_stones(point, self._next_player)) != 0:
                        back()
                        raise GoIllegalActionError.commit_suicide(self._next_player, point, self)
                    self._next_player = player.other
                    return back
                else:  # self._grid.item(point) != GoPlayer.none.value
                    raise GoIllegalActionError.already_has_a_stone(self._next_player, point, self)
            except IndexError:
                raise GoIllegalActionError.move_out_of_range(self._next_player, point, self)
