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

    def _retrieve_one(self, url: str, filename: Optional[str] = None, force=False):
        if filename is None:
            filename = path.basename(url)
        filename = path.join(self.archive_dir, filename)
        if force or not path.exists(filename):
            tmp_name = filename + ".download"
            urlretrieve(url, tmp_name)
            rename(tmp_name, filename)

    def retrieve(self, force=False):
        print("Preparing to download archives...")
        self._retrieve_one("http://u-go.net/gamerecords/", "kgs_index.html", force)
        kgs_index = path.join(self.archive_dir, "kgs_index.html")
        with open(kgs_index, "r") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            links = soup.select("body>div>div:nth-child(2)>div.col-md-10>table>tr>td:nth-child(6)>a")
            urls: List[str] = []
            for link in links:
                archive_url: str = link.get("href")
                if archive_url.endswith(".tar.gz"):
                    urls.append(archive_url)
            do_works(partial(self._retrieve_one, filename=None, force=force), urls, desc="Datasets downloading...",
                     unit="archive", cpu=False)

