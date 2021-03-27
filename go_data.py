#!/usr/bin/env python
# -*- coding: utf-8 -*-

from urllib.request import urlretrieve
from bs4 import BeautifulSoup
from os import rename, path, cpu_count
from glob import glob
from sgfmill import sgf
import tarfile
from typing import *
import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from go_cache import *
from functools import partial

__all__ = ["UGoArchive"]


class UGoArchive(GameArchive):

    @staticmethod
    def _retrieve_one(url: str, filename: Optional[str] = None, force: bool = False):
        if filename is None:
            filename = path.basename(url)
        filename = path.join(get_archive_dir(), filename)
        if force or not path.exists(filename):
            tmp_name = filename + ".download"
            urlretrieve(url, tmp_name)
            rename(tmp_name, filename)

    @staticmethod
    def retrieve(force: bool = False):
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
            with ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
                tasks = executor.map(partial(UGoArchive._retrieve_one, filename=None, force=force), urls)
                for _ in tqdm.tqdm(tasks, total=len(urls), desc="Downloading...", unit="archive"):
                    pass

    @staticmethod
    def _unpack_one(archive: str, force: bool = False):
        dest_path = path.join(get_archive_dir(), path.splitext(archive)[0])
        tmp_path = dest_path + ".tmp"
        if force or not path.exists(dest_path):
            with tarfile.open(archive) as a:
                a.extractall(tmp_path)
                rename(tmp_path, dest_path)

    @staticmethod
    def unpack(force=False):
        archives = list(glob(path.join(get_archive_dir(), "*.tar.gz")))
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            tasks = executor.map(partial(UGoArchive._unpack_one, force=force), archives)
            for _ in tqdm.tqdm(tasks, total=len(archives), desc="Unpacking...", unit="archive"):
                pass

    @staticmethod
    def _extract_one(file_name: str, force=False):
        name = path.splitext(path.basename(file_name))[0]
        with open(file_name, "rb") as f:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())
            if force or not GameData.pickle_exists(name, sgf_game.size):
                game_data = GameData.from_sgf(sgf_game)
                if len(game_data.sequence) > 1:
                    game_data.to_pickle(name)

    @staticmethod
    def extract(force=False):
        files = glob(path.join(get_archive_dir(), "**/*.sgf"), recursive=True)
        bar = tqdm.tqdm(total=len(files), desc="Extracting...", unit="file", unit_scale=True)
        with ProcessPoolExecutor(max_workers=cpu_count() // 2) as executor:
            while len(files) > 0:
                processing_files = files[:10000]
                files = files[10000:]
                results = executor.map(partial(UGoArchive._extract_one, force=force), processing_files)
                for _ in results:
                    bar.update(1)

    def download(self, force=False):
        self.retrieve(force=force)
        self.unpack(force=force)
        self.extract(force=force)


if __name__ == "__main__":
    set_cache_dir()
    ar = UGoArchive()
    ar.download()
