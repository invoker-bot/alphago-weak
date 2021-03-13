# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 3/12/2021
@Version : 1.0
"""

import asyncio
import sys
import re

def init():
    """
    Set compatible event loop for async I/O stream.
    As of Python < 3.8, the default event loop on Windows is `WindowsSelectorEventLoopPolicy`.
    """
    if sys.platform.startswith("win") and sys.version_info < (3, 8) and isinstance(asyncio.get_event_loop_policy(),
                                                                                   asyncio.WindowsSelectorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


class GTPServer:

    def __init__(self, cmd: str):
        self.cmd = cmd
        self.proc = None

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(self.cmd, stdin=asyncio.subprocess.PIPE,
                                                         stdout=asyncio.subprocess.PIPE)
        self.proc.stdin.write(b"")


    def parse_response(self,cmd:str):
        re.compile(r"^\s* = (?P<id>\d+)? (?P<response>.*)",re.VERBOSE)

init()

if __name__ == '__main__':
    pass
