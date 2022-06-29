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
import tensorflow as tf
from os import path, makedirs, rename
from glob import glob, iglob
from functools import partial
from sgfmill import sgf
from sgfmill.sgf import Sgf_game
from typing import *
from ..board import *
from ..utils.multi_works import do_works

__all__ = ["GameData", "GameArchive"]


class GameData(NamedTuple):
    size: int
    winner: GoPlayer
    sequence: List[Tuple[Optional[GoPlayer], Optional[Union[GoPoint, Tuple[int, int]]]]]
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
        return cls(size=size, winner=winner, sequence=sequence, komi=komi, setup_stones=setup_stones)


class GameArchive(object):

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
            with open(file_name, "rb") as f:
                sgf_game = sgf.Sgf_game.from_bytes(f.read())
                game_data = GameData.from_sgf(sgf_game)
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
        for game_data_path in iglob(self.data_pattern):
            with open(game_data_path, "rb") as f:
                yield pickle.load(f)

    def __len__(self):
        return len(glob(self.data_pattern))

    def main(self, args: Sequence[str] = None):
        parser = argparse.ArgumentParser(description="useful for downloading archives")
        parser.add_argument("-r", "--root", default=None, help="root dir for caching")
        parser.add_argument("-f", "--force", default=False, type=bool, help="force to re-download")
        res = parser.parse_args(args)
        if res.root is not None:
            self.root = res.root
        self.download(res.force)