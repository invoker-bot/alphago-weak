# -*- coding: utf-8 -*-

from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import os
from os import path, cpu_count
from glob import glob
from sgfmill import sgf
import tarfile
from typing import *
import tqdm
from functools import partial

from ..basic import *

__all__ = ["UGoArchive"]


def rename(src: str, dst: str):
    if path.exists(dst):
        os.remove(dst)
    os.rename(src, dst)


class UGoArchive(GameArchive):
    name = "u-go"

    @staticmethod
    def _retrieve_one(url: str, filename: Optional[str] = None, force=False):
        if filename is None:
            filename = path.basename(url)
        filename = path.join(get_archive_dir(), filename)
        if force or not path.exists(filename):
            tmp_name = filename + ".download"
            urlretrieve(url, tmp_name)
            rename(tmp_name, filename)

    def retrieve(self, force=False):
        tqdm.tqdm.write("Preparing to download...")
        UGoArchive._retrieve_one('http://u-go.net/gamerecords/', "kgs_index.html", force)
        kgs_index = path.join(get_archive_dir(), "kgs_index.html")
        with open(kgs_index, "r") as f:
            soup = BeautifulSoup(f.read(), "lxml")
            links = soup.select('body>div>div:nth-child(2)>div.col-md-10>table>tr>td:nth-child(6)>a')
            urls: List[str] = []
            for link in links:
                archive_url: str = link.get("href")
                if archive_url.endswith('.tar.gz'):
                    urls.append(archive_url)
            do_works(partial(UGoArchive._retrieve_one, filename=None, force=force), urls, desc="Downloading...",
                     unit="archive", cpu=False)

    @staticmethod
    def _unpack_one(archive: str, force=False):
        dest_path = path.join(get_archive_dir(), path.splitext(archive)[0])
        tmp_path = dest_path + ".tmp"
        if force or not path.exists(dest_path):
            with tarfile.open(archive) as a:
                a.extractall(tmp_path)
                rename(tmp_path, dest_path)

    def unpack(self, force=False):
        archives = list(glob(path.join(get_archive_dir(), "*.tar.gz")))
        do_works(partial(UGoArchive._unpack_one, force=force), archives, desc="Unpacking", unit="archive")

    @staticmethod
    def _extract_one(file_name: str, force=False):
        name = path.splitext(path.basename(file_name))[0]
        with open(file_name, "rb") as f:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())
            if force or not GameData.pickle_exists(name, sgf_game.size):
                game_data = GameData.from_sgf(sgf_game)
                if len(game_data.sequence) > 1:
                    game_data.to_pickle(name)

    def extract(self, force=False):
        files = glob(path.join(get_archive_dir(), "**/*.sgf"), recursive=True)
        do_works(partial(UGoArchive._extract_one, force=force), files, desc="Extracting", unit="file")
