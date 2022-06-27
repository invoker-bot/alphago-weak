# -*- coding: utf-8 -*-

from os import path, getcwd, makedirs, listdir, remove
from typing import *
import pickle
from abc import ABCMeta, abstractmethod
from sgfmill.sgf import Sgf_game
import numpy as np

from .go_types import *

__all__ = ["set_cache_dir", "get_cache_dir", "get_game_dir", "get_archive_dir",
           "get_array_dir", "GameData", "GameArchive",
           "GameDatabase", "ArrayDatabase"]

default_cache_dir = path.join(path.dirname(path.realpath(__file__)), "../..", ".data")
cache_dir = default_cache_dir
archive_folder = path.join(cache_dir, ".kgs")
game_folder = path.join(cache_dir, ".game")
array_folder = path.join(cache_dir, ".array")


def set_cache_dir(directory: Optional[str] = None) -> NoReturn:
    global cache_dir, archive_folder, game_folder, array_folder
    if directory is None:
        directory = default_cache_dir
    cache_dir = path.join(getcwd(), directory)
    archive_folder = path.join(cache_dir, ".kgs")
    game_folder = path.join(cache_dir, ".game")
    array_folder = path.join(cache_dir, ".array")
    makedirs(get_cache_dir(), exist_ok=True)
    makedirs(get_archive_dir(), exist_ok=True)
    makedirs(get_game_dir(), exist_ok=True)
    makedirs(get_array_dir(), exist_ok=True)


def get_cache_dir() -> str:
    return cache_dir


def get_archive_dir() -> str:
    return archive_folder


def get_game_dir() -> str:
    return game_folder


def get_array_dir() -> str:
    return array_folder


class GameData(NamedTuple):
    size: int
    winner: GoPlayer
    sequence: List[Tuple[Optional[GoPlayer], Optional[GoPoint]]]
    komi: float
    setup_stones: Tuple[Optional[Set[GoPoint]], Optional[Set[GoPoint]], Optional[Set[GoPoint]]]

    @classmethod
    def from_sgf(cls, sgf_game: Sgf_game):
        size = sgf_game.get_size()
        winner = GoPlayer.to_player(sgf_game.get_winner())
        sequence = list(map(lambda move: (GoPlayer.to_player(move[0]), move[1]),
                            (node.get_move() for node in sgf_game.get_main_sequence())))
        komi = sgf_game.get_komi()
        setup_stones = sgf_game.get_root().get_setup_stones()
        return cls(size, winner, sequence, komi, setup_stones)

    @staticmethod
    def from_pickle(name: str, size: Union[int, str] = 19):
        with open(path.join(get_game_dir(), str(size), name), "rb") as f:
            return pickle.load(f)

    @staticmethod
    def pickle_exists(name: str, size: Union[int, str] = 19):
        return path.exists(path.join(get_game_dir(), str(size), name))

    def to_pickle(self, name: str):
        makedirs(self.root(), exist_ok=True)
        dest = self.path(name)
        with open(dest, "wb") as f:
            pickle.dump(self, f)

    def root(self):
        return path.join(get_game_dir(), str(self.size))

    def path(self, name: str):
        return path.join(self.root(), name)


class GameArchive(metaclass=ABCMeta):
    name = "none"

    @classmethod
    def archive_map(cls):
        _dict = {_cls.name: _cls for _cls in cls.__subclasses__()}
        for v in cls.__subclasses__():
            _dict.update(v.archive_map())
        return _dict

    @abstractmethod
    def retrieve(self, force=False) -> NoReturn:
        """
        Retrieve all archives available from Internet.
        :param force: whether forces to dataset archive if it has already existed
        """
        pass

    @abstractmethod
    def extract(self, force=False) -> NoReturn:
        """
        Extract all game archives to Game Cache Folder, every single file should end with `.game.pkl` and be
        start with it's size of the board.
        """
        pass

    @abstractmethod
    def unpack(self, force=False) -> NoReturn:
        """
        Unpack all game archives to
        :param force: whether forces to dataset archive if it has already existed
        """
        pass

    def download(self, force=False):
        self.retrieve(force=force)
        self.unpack(force=force)
        self.extract(force=force)


class GameDatabase:

    def __init__(self, size=19):
        self.size = size

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, name: str) -> GameData:
        return GameData.from_pickle(name, self.size)

    def __setitem__(self, name: str, data: GameData):
        data.to_pickle(name)

    def __delitem__(self, name: str):
        remove(path.join(get_game_dir(), str(self.size), name))

    def __contains__(self, name: str):
        return path.exists(path.join(get_game_dir(), str(self.size), name))

    def __eq__(self, other):
        if isinstance(other, GameDatabase):
            return self.size == other.size
        return NotImplemented

    def root(self):
        return path.join(get_game_dir(), str(self.size))

    def keys(self) -> List[str]:
        return listdir(self.root())

    def values(self) -> Iterable[GameData]:
        for key in self.keys():
            yield self[key]

    def items(self) -> Iterable[Tuple[str, GameData]]:
        for key in self.keys():
            yield key, self[key]


class ArrayDatabase:

    def __init__(self, method: str, size=19):
        self.size = size
        self.method = method
        makedirs(self.root(), exist_ok=True)

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, key: str) -> Tuple[np.ndarray, ...]:
        file = path.join(self.root(), key)
        with open(file, "rb") as f:
            return pickle.load(f)

    def __setitem__(self, key: str, value: Tuple[np.ndarray, ...]):
        file = path.join(self.root(), key)
        with open(file, "wb") as f:
            pickle.dump(value, f)

    def __delitem__(self, key: str):
        file = path.join(self.root(), key)
        remove(file)

    def __contains__(self, key: str):
        return path.exists(path.join(self.root(), key))

    def root(self):
        return path.join(get_array_dir(), str(self.size), self.method)

    def keys(self) -> List[str]:
        return listdir(self.root())

    def values(self) -> Iterable[np.ndarray]:
        for key in self.keys():
            yield self[key]

    def items(self) -> Iterable[Tuple[str, np.ndarray]]:
        for key in self.keys():
            yield key, self[key]
