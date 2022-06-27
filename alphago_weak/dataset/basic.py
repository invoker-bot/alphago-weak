# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2021/11/19
@Version : 1.0
"""

from os import path, makedirs
from sgfmill.sgf import Sgf_game
from abc import *
from typing import *
from ..board import *

__all__ = ["GameData", "GameArchive"]


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
        return cls("GameData", size=size, winner=winner, sequence=sequence, komi=komi, setup_stones=setup_stones)


class GameArchive(metaclass=ABCMeta):

    def __init__(self, root: str = None):
        if root is None:
            root = path.join(path.dirname(path.realpath(__file__)), "../..", ".data")
        self.root = root
        makedirs(self.root, exist_ok=True)
        makedirs(self.archive_dir, exist_ok=True)
        makedirs(self.data_dir, exist_ok=True)

    @property
    def archive_dir(self):
        return path.join(self.root, "archive")

    @property
    def data_dir(self):
        return path.join(self.root, "data")

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
