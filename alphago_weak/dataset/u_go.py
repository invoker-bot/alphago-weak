# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2022/6/27
@Version : 1.1
"""

import asyncio
import os
from os import path
from urllib.parse import urljoin, urlparse
from typing import *

import aiohttp
from bs4 import BeautifulSoup
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from .basic import *

__all__ = ["UGoArchive"]

INDEX_URL = "http://u-go.net/gamerecords/"
DOWNLOAD_CONCURRENCY = 8
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def download_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def rename(src: str, dst: str):
    if path.exists(dst):
        os.remove(dst)
    os.rename(src, dst)


class UGoArchive(GameArchive):

    def _archive_urls_from_index(self, kgs_index: str) -> List[str]:
        with open(kgs_index, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        urls: List[str] = []
        links = soup.select("body>div>div:nth-child(2)>div.col-md-10>table>tr>td:nth-child(6)>a")
        for link in links:
            archive_url = urljoin(INDEX_URL, link.get("href", ""))
            if archive_url.endswith(".tar.gz"):
                urls.append(archive_url)
        return urls

    async def _retrieve_one_async(self, session: aiohttp.ClientSession, url: str,
                                  filename: Optional[str] = None, force=False) -> str:
        if filename is None:
            filename = path.basename(urlparse(url).path)
        if not filename:
            raise ValueError(f"cannot infer filename from url: {url}")
        filename = path.join(self.archive_dir, filename)
        if not force and path.exists(filename):
            return filename

        tmp_name = filename + ".download"
        if path.exists(tmp_name):
            os.remove(tmp_name)

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(tmp_name, "wb") as f:
                    async for chunk in response.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            rename(tmp_name, filename)
        except Exception:
            if path.exists(tmp_name):
                os.remove(tmp_name)
            raise
        return filename

    async def retrieve_async(self, force=False):
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)
        connector = aiohttp.TCPConnector(limit=DOWNLOAD_CONCURRENCY, resolver=aiohttp.ThreadedResolver())
        headers = {"User-Agent": "AlphaGoWeak/1.0"}
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers, trust_env=True) as session:
            kgs_index = await self._retrieve_one_async(session, INDEX_URL, "kgs_index.html", force)
            urls = self._archive_urls_from_index(kgs_index)
            semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

            async def retrieve_archive(url: str):
                async with semaphore:
                    return await self._retrieve_one_async(session, url, force=force)

            tasks = [asyncio.create_task(retrieve_archive(url)) for url in urls]
            with download_progress() as progress:
                progress_task = progress.add_task("Datasets downloading... (archive)", total=len(tasks))
                for task in asyncio.as_completed(tasks):
                    await task
                    progress.advance(progress_task)

    def retrieve(self, force=False):
        print("Preparing to download archives...")
        asyncio.run(self.retrieve_async(force))

