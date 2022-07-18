# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2022/6/28
@Version : 1.1
"""

import tarfile
from importlib import import_module
from os import path, makedirs, rename
from glob import glob, iglob
from itertools import starmap
from functools import partial
from sgfmill.sgf import Sgf_game
from typing import *
from ..board import *
from ..utils.multi_works import do_works_experimental

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

    def __init__(self, root: str):
        self._root = root
        makedirs(self.archive_dir, exist_ok=True)
        self._data_files = glob(self.data_pattern, recursive=True)

    @property
    def archive_dir(self):
        return path.join(self._root, "archive")

    @property
    def cache_dir(self):
        return path.join(self._root, "cache")

    def retrieve(self, force=False):
        """
        Retrieve all go game archives from the Internet.
        :param force: whether forces to retrieve dataset archives if they have already existed
        """
        pass

    def _unpack_one(self, archive: str, force=False):
        dest_path = path.join(self.archive_dir, path.splitext(path.basename(archive))[0])
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
        do_works_experimental(partial(self._unpack_one, force=force), archives, desc="Unpacking", unit="archive")

    def download(self, force=False):
        self.retrieve(force=force)
        self.unpack(force=force)

    @property
    def data_pattern(self):
        return path.join(self.archive_dir, "**/*.sgf")

    def __iter__(self) -> Iterator[GameData]:
        return map(GameData.from_sgf, self._data_files)

    def __getitem__(self, item: int) -> GameData:
        return GameData.from_sgf(self._data_files[item])

    def __len__(self):
        return len(self._data_files)
