import numpy as np
from itertools import chain, product
from typing import *
from go_types import *

__all__ = [
    "GoBoard"
]


class GoBoard(GoBoardBase):

    def __init__(self, shape: int = 19, first_player=GoPlayer.black):
        super().__init__(shape, first_player)
        self._hash = 0
        self.__cache = []

    def push_state(self):
        self.__cache.append((self._hash, self._grid.copy(), self._next_player))

    def pop_state(self):
        self._hash, self._grid, self._next_player = self.__cache.pop()

    def __place_stone(self, stone: GoPoint, player: GoPlayer = GoPlayer.none):
        self._grid.itemset(stone, player)

    def __place_stones(self, stones: Iterable[GoPoint], player: GoPlayer = GoPlayer.none):
        for stone in stones:
            self.__place_stone(stone, player)

    def _get_neighbors(self, point: GoPoint) -> Tuple[GoPoint]:
        shape = self._grid.shape
        x, y = point
        return tuple((px, py) for px, py in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)) if
                     0 <= px < shape[0] and 0 <= py < shape[1])

    def get_string(self, point: GoPoint) -> GoString:
        assert 0 <= point[0] < self._grid.shape[0] and 0 <= point[1] < self._grid.shape[1]
        player = self._grid.item(point)
        string = GoString(GoPlayer(player), set(), set())
        if player == GoPlayer.none:
            string.liberties.add(point)
            return string
        else:
            string.stones.add(point)
            neighbors_queue = set(self._get_neighbors(point))
            while len(neighbors_queue) != 0:
                point = neighbors_queue.pop()
                point_t = self._grid.item(point)
                if point_t == GoPlayer.none:  # liberty
                    string.liberties.add(point)
                elif point_t == player and point not in string.stones:  # stone
                    string.stones.add(point)
                    neighbors_queue.update(self._get_neighbors(point))
            return string

    def is_valid_point(self, point: GoPoint) -> bool:
        try:
            return self.move(point, check=True)
        except GoIllegalActionError:
            return False

    def __may_dead(self, point: GoPoint) -> bool:
        neighbors = self._get_neighbors(point)
        return any(map(lambda n: self._grid.item(n) != 0, neighbors))

    def __remove_if_dead(self, point: GoPoint) -> bool:
        string = self.get_string(point)
        if len(string.liberties) == 0:
            self.__place_stones(string.stones, 0)
            return True
        return False

    def __get_dead_stones(self, point: GoPoint) -> Set[GoPoint]:
        string = self.get_string(point)
        if len(string.liberties) == 0:
            return string.stones
        return set()

    def move(self, point: Optional[GoPoint] = None, check=False) -> bool:
        player = self._next_player
        if point is not None:
            try:
                if self._grid.item(point) == GoPlayer.none:
                    if self.__may_dead(point):
                        self.push_state()
                        self.__place_stone(point, player)  # try place
                        _dead_stones = set()
                        for neighbor in self._get_neighbors(point):
                            if self._grid.item(neighbor) == player.other:
                                _dead_stones.update(self.__get_dead_stones(neighbor))
                        self.__place_stones(_dead_stones)
                        for neighbor in chain(self._get_neighbors(point), (point,)):
                            if self._grid.item(neighbor) == player:
                                if len(self.__get_dead_stones(neighbor)) > 0:
                                    self.pop_state()
                                    raise GoIllegalActionError.commit_suicide(player, point, self)
                        if check:
                            self.pop_state()
                        else:
                            self.__cache.pop()
                    else:
                        if not check:
                            self.__place_stone(point, player)
                else:
                    raise GoIllegalActionError.already_has_a_stone(player, point, self)
            except IndexError as e:
                raise GoIllegalActionError.move_out_of_range(player, point, self)
        if not check:
            self._next_player = player.other
        return True

    def play(self, point: Optional[GoPoint] = None):
        self.move(point, False)

    def is_point_a_fake_eye(self, point: GoPoint) -> bool:
        return self._grid.item(point) == GoPlayer.none and \
               (all(map(lambda n: self._grid.item(n) == GoPlayer.white, self._get_neighbors(point))) or \
                all(map(lambda n: self._grid.item(n) == GoPlayer.black, self._get_neighbors(point))))

    @property
    def shape(self) -> Tuple[int, int]:
        return self._grid.shape

    def grid_tensor(self, dtype=np.float32):
        shape = self._grid.shape
        tensor = np.zeros(shape=(2, shape[0], shape[1]), dtype=dtype)
        player = self._next_player
        for x, y in product(range(shape[0]), range(shape[1])):
            stone = self._grid.item((x, y))
            if stone == player:  # 等于next_player时为1
                tensor.itemset((0, x, y), 1.0)
            elif stone == 0:
                tensor.itemset((0, x, y), 0.5)
            if self.is_valid_point((x, y)):
                tensor.itemset((1, x, y), 1.0)
        return tensor

    def encode_point(self, point: Optional[GoPoint]) -> int:
        shape = self._grid.shape
        if point:
            return point[0] * shape[0] + point[1]
        return shape[0] * shape[1]

    def decode_point(self, number: int) -> Optional[GoPoint]:
        shape = self._grid.shape
        if number == shape[0] * shape[1]:
            return None
        return divmod(number, shape[0])
