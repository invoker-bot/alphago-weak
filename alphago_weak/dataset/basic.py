# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2022/6/28
@Version : 1.1
"""

import pickle
import tarfile
import argparse
from importlib import import_module
from os import path, makedirs, rename
from glob import glob, iglob
from itertools import starmap
from functools import partial
from sgfmill.sgf import Sgf_game
from typing import *
from ..board import *
from ..utils.multi_works import do_works

__all__ = ["GameData", "GameArchive"]


class GameData(NamedTuple):
    size: int = 19
    winner: GoPlayer = GoPlayer.none
    sequence: List[Tuple[Optional[GoPlayer], Optional[Union[GoPoint, Tuple[int, int]]]]] = []
    komi: float = 6.5
    setup_stones: Tuple[Set[GoPoint], Set[GoPoint], Set[GoPoint]] = (set(), set(), set())

    @staticmethod
    def from_sgf(sgf_file: str):
        with open(sgf_file, "rb") as f:
            sgf_game = Sgf_game.from_bytes(f.read())
        size = sgf_game.get_size()
        winner = GoPlayer.to_player(sgf_game.get_winner())
        sequence = list(map(lambda move: (GoPlayer.to_player(move[0]), GoPoint.to_point(move[1])),
                            (node.get_move() for node in sgf_game.get_main_sequence())))
        komi = sgf_game.get_komi()
        setup_stones: Any = tuple(set(starmap(GoPoint, points)) for points in sgf_game.get_root().get_setup_stones())
        return GameData(size, winner, sequence, komi, setup_stones)

    def to_sgf(self, sgf_file: str):
        sgf_game = Sgf_game(size=self.size)
        sgf_game.set_date()
        root = sgf_game.get_root()
        # Player name PB PW
        # Comment AN, C
        # PC name
        root.set("PC", "Desktop")
        # komi
        root.set("KM", self.komi)
        # winner
        root.set("RE", self.winner.to_sgf())
        # handicap
        root.set("HA", len(self.setup_stones[0]) - len(self.setup_stones[1]))
        for identifier, value in zip(("AB", "AW", "AE"), set(map(tuple, self.setup_stones))):
            if len(value) > 0:
                root.set(identifier, value)

        for player, point in self.sequence:
            if player != GoPlayer.none:
                node = sgf_game.extend_main_sequence()
                node.set_move(player.to_sgf(), tuple(point) if point is not None else None)
        with open(sgf_file, "wb") as f:
            f.write(sgf_game.serialise())


PKG = "alphago_weak.dataset"


class GameArchive(object):
    FACTORY_DICT = {
        "u_go": lambda root: import_module(".u_go", PKG).UGoArchive(root),
    }

    def __init__(self, root: str = None):
        if root is None:
            root = path.join(path.dirname(path.realpath(__file__)), "../..", ".data")
        self._root = root
        makedirs(self._root, exist_ok=True)
        makedirs(self.archive_dir, exist_ok=True)
        makedirs(self.data_dir, exist_ok=True)
        self._data_files = glob(self.data_pattern)

    @property
    def archive_dir(self):
        return path.join(self._root, "archive")

    @property
    def data_dir(self):
        return path.join(self._root, "data")

    def retrieve(self, force=False):
        """
        Retrieve all go game archives from the Internet.
        :param force: whether forces to retrieve dataset archives if they have already existed
        """
        pass

    def _unpack_one(self, archive: str, force=False):
        dest_path = path.join(self.archive_dir, path.splitext(archive)[0])
        tmp_path = dest_path + ".tmp"
        if force or not path.exists(dest_path):
            with tarfile.open(archive) as a:
                a.extractall(tmp_path)
                rename(tmp_path, dest_path)

    def unpack(self, force=False):
        """Unpack all game archives to common file formats (e.g. sgf, etc.).
        :param force: whether forces to unpack dataset archives if they have already unpacked
        """
        print("Preparing to unpack downloaded archives...")
        archives = list(glob(path.join(self.archive_dir, "*.tar.gz")))
        do_works(partial(self._unpack_one, force=force), archives, desc="Unpacking", unit="archive")

    def _extract_one(self, file_name: str, force=True):
        name = path.splitext(path.basename(file_name))[0]
        data_path = path.join(self.data_dir, name + ".gamedata")
        if force or not path.exists(data_path):
            game_data = GameData.from_sgf(file_name)
            if len(game_data.sequence) > 1:
                with open(data_path, "wb") as data_f:
                    pickle.dump(game_data, data_f)

    def extract(self, force=True):
        """Extract all game unpacked archives to Game Data Folder, every single game data file should
            end with `.gamedata` and be named with it's size of the board.
        """
        files = glob(path.join(self.archive_dir, "**/*.sgf"), recursive=True)
        do_works(partial(self._extract_one, force=force), files, desc="Extracting", unit="file")

    def download(self, force=False):
        self.retrieve(force=force)
        self.unpack(force=force)
        self.extract(force=force)

    @property
    def data_pattern(self):
        return path.join(self.data_dir, "*.gamedata")

    def __iter__(self) -> Iterator[GameData]:
        for game_data_path in self._data_files:
            with open(game_data_path, "rb") as f:
                yield pickle.load(f)

    def __getitem__(self, item: int) -> GameData:
        with open(self._data_files[item], "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return len(self._data_files)

