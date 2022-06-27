# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2022/6/27
@Version : 1.1
"""

from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import os
from os import path
from glob import glob
from sgfmill import sgf
import tarfile
import pickle
from functools import partial
from typing import *
from .basic import *
from ..utils.multi_works import do_works

__all__ = ["UGoArchive"]


def rename(src: str, dst: str):
    if path.exists(dst):
        os.remove(dst)
    os.rename(src, dst)


class UGoArchive(GameArchive):
    name = "u-go"

    def _retrieve_one(self, url: str, filename: Optional[str] = None, force=False):
        if filename is None:
            filename = path.basename(url)
        filename = path.join(self.archive_dir, filename)
        if force or not path.exists(filename):
            tmp_name = filename + ".download"
            urlretrieve(url, tmp_name)
            rename(tmp_name, filename)

    def retrieve(self, force=False):
        print("Preparing to download datasets...")
        self._retrieve_one("http://u-go.net/gamerecords/", "kgs_index.html", force)
        kgs_index = path.join(self.archive_dir, "kgs_index.html")
        with open(kgs_index, "r") as f:
            soup = BeautifulSoup(f.read(), "lxml")
            links = soup.select("body>div>div:nth-child(2)>div.col-md-10>table>tr>td:nth-child(6)>a")
            urls: List[str] = []
            for link in links:
                archive_url: str = link.get("href")
                if archive_url.endswith(".tar.gz"):
                    urls.append(archive_url)
            do_works(partial(self._retrieve_one, filename=None, force=force), urls, desc="Datasets downloading...",
                     unit="archive", cpu=False)

    def _unpack_one(self, archive: str, force=False):
        dest_path = path.join(self.archive_dir, path.splitext(archive)[0])
        tmp_path = dest_path + ".tmp"
        if force or not path.exists(dest_path):
            with tarfile.open(archive) as a:
                a.extractall(tmp_path)
                rename(tmp_path, dest_path)

    def unpack(self, force=False):
        archives = list(glob(path.join(self.archive_dir, "*.tar.gz")))
        do_works(partial(self._unpack_one, force=force), archives, desc="Unpacking", unit="archive")

    def _extract_one(self, file_name: str, force=False):
        name = path.splitext(path.basename(file_name))[0]
        with open(file_name, "rb") as f:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())
            data_path = path.join(self.data_dir, sgf_game.size, name + ".data")
            if force or not path.exists(data_path):
                game_data = GameData.from_sgf(sgf_game)
                if len(game_data.sequence) > 1:
                    with open(data_path, "wb") as data_f:
                        pickle.dump(game_data, data_f)

    def extract(self, force=False):
        files = glob(path.join(self.archive_dir, "**/*.sgf"), recursive=True)
        do_works(partial(self._extract_one, force=force), files, desc="Extracting", unit="file")
