from enum import IntEnum
from typing import Tuple, NamedTuple, Union, Set, Iterable, Optional, Any, Callable, NoReturn, Iterator
import numpy as np
from collections import Counter
from colorama import init, Fore, Back, Style
from abc import abstractmethod, ABCMeta
from itertools import product

init(autoreset=True)


class GoPlayer(IntEnum):
    black = 0
    white = 1
    none = 2

    @property
    def other(self):
        if self is GoPlayer.none:
            return self
        return GoPlayer.white if self is GoPlayer.black else GoPlayer.black

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

    def __init__(self, shape: int = 19, first_player=GoPlayer.black):
        self._next_player = first_player
        self._grid = np.full((shape, shape), GoPlayer.none.value, dtype=np.uint8)

    def setup_stones(self, black_stones: Optional[Iterable[GoPoint]] = None,
                     white_stones: Optional[Iterable[GoPoint]] = None,
                     empty_stones: Optional[Iterable[GoPoint]] = None) -> NoReturn:
        if black_stones:
            for stone in black_stones:
                self._grid.itemset(stone, 1)
        if white_stones:
            for stone in white_stones:
                self._grid.itemset(stone, -1)

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

    @abstractmethod
    def get_string(self, point: GoPoint) -> Optional[GoString]:
        pass

    @abstractmethod
    def is_valid_point(self, point: GoPoint) -> bool:
        pass

    @abstractmethod
    def play(self, point: Optional[GoPoint] = None) -> Any:
        pass

    @abstractmethod
    def is_point_a_fake_eye(self, point: GoPoint) -> bool:
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
                    cols.append(" ")
                elif color == GoPlayer.white.value:
                    cols.append(Fore.WHITE + "o" + Fore.RESET)
                else:
                    cols.append(Fore.BLACK + 'x' + Fore.RESET)
            cols.append("")
            rows.append((Fore.LIGHTBLACK_EX + '|' + Fore.RESET).join(cols))
        rows.append("")
        return '\n'.join(rows)

    def __str__(self):
        return self.details()

    '''
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

    def encode_point(self, point: Optional[Point]) -> int:
        shape = self._grid.shape
        if point:
            return point[0] * shape[0] + point[1]
        return shape[0] * shape[1]

    def decode_point(self, number: int) -> Optional[Point]:
        shape = self._grid.shape
        if number == shape[0] * shape[1]:
            return None
        return divmod(number, shape[0])
    '''


class GoPlayerControllerBase(metaclass=ABCMeta):

    def __init__(self, board: GoBoardBase, player: GoPlayer):
        self.board = board
        self.player = player

    @abstractmethod
    def play(self):
        pass


class GoBotBase(metaclass=ABCMeta):

    def __init__(self, board: GoBoardBase, player: GoPlayer):
        self.board = board
        self.player = player

    @abstractmethod
    def play(self):
        pass
