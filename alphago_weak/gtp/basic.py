# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2022/6/29
@Version : 1.1
"""

import sys
import re
import string
from importlib import import_module
from abc import *
from typing import *

from ..board import *

__all__ = ["GTPClient"]

PKG = "alphago_weak.gtp"

class GTPClient(metaclass=ABCMeta):
    __version__ = "1.0"
    name: str = None
    FACTORY_DICT = {
        "random_bot": lambda: import_module(".gtp_random_bot", PKG).GTPRandomBot(),
        "random_bot_mcts": lambda: import_module(".gtp_random_bot_mcts", PKG).GTPRandomBotMCTS(),
    }

    CMD = re.compile(r"^\s* (?P<id>\d+)? \s* (?P<command>\w*) \s* (?P<args>.*)", re.VERBOSE)
    MOVE = re.compile(r"^\s*(?P<x>[abcdefghjklmnopqrst])(?P<y>\d{1,2})", re.VERBOSE | re.IGNORECASE)
    player = re.compile(r"^\s* (?P<player>w|b|white|black) \s* (?P<move>\w+)?", re.VERBOSE | re.IGNORECASE)

    COORDINATE = tuple("ABCDEFGHJKLMNOPQRSTUVWXYZ")
    COORDINATE_R = {coor: idx for idx, coor in enumerate(COORDINATE)}

    def __init__(self):
        self.komi = 6.5
        self.config = {}

    @staticmethod
    def do_protocol_version(_id: str = "", args: str = ""):
        print("=", _id, 2, "\n", flush=True)

    def do_name(self, _id: str = "", args: str = ""):
        print("=", _id, string.capwords(self.name.replace("_", " ")), "\n", flush=True)

    def do_version(self, _id: str = "", args: str = ""):
        print("=", _id, self.__version__, "\n", flush=True)

    def do_known_command(self, _id: str = "", args: str = ""):
        result = "true" if hasattr(self, "do_" + args.strip()) else "false"
        print("=", _id, result, "\n", flush=True)

    def do_list_commands(self, _id: str = "", args: str = ""):
        cmds = [attr[3:] for attr in dir(self) if attr.startswith("do_")]
        print("=", _id, "\n".join(cmds), "\n", flush=True)

    @staticmethod
    def do_quit(_id: str = "", args: str = ""):
        print("=", _id, "\n", flush=True)
        sys.exit(0)

    def do_boardsize(self, _id: str = "", args: str = ""):
        try:
            size = int(args)
            if 0 < size <= 25 and self.boardsize(size):
                print("=", _id, "\n", flush=True)
            else:
                print("?", _id, "unacceptable size", "\n", flush=True)
        except ValueError:
            print("?", _id, "boardsize is not an integer", "\n", flush=True)

    def do_komi(self, _id: str = "", args: str = ""):
        try:
            self.komi = float(args)
            print("=", _id, "\n", flush=True)
        except ValueError:
            print("?", _id, "komi not a float", "\n", flush=True)

    def do_clear_board(self, _id: str = "", args: str = ""):
        self.clear_board()
        print("=", _id, "\n", flush=True)

    def do_play(self, _id: str = "", args: str = ""):
        parsed = self.player.match(args)
        if parsed and parsed["move"]:
            player = parsed["player"].lower()
            action = parsed["move"].strip().lower()
            if action == "pass" or action == "resign":
                print("=", _id, "\n", flush=True)
            else:
                move_parsed = self.MOVE.match(action)
                x = self.COORDINATE_R[move_parsed["x"].upper()]
                y = int(move_parsed["y"]) - 1
                if self.play(GoPlayer.to_player(player), GoPoint(x, y)):
                    print("=", _id, "\n", flush=True)
                else:
                    print("?", _id, "illegal move", "\n", flush=True)
        else:
            print("?", _id, "invalid player or coordinate", "\n", flush=True)

    def do_genmove(self, _id: str = "", args: str = ""):
        parsed = self.player.match(args)
        if parsed:
            player = GoPlayer.to_player(parsed["player"].lower())
            result = self.genmove(player)
            if not isinstance(result, str):
                self.play(player, result)
                result = self.COORDINATE[result.x] + str(result.y + 1)
            print("=", _id, result, "\n", flush=True)
        else:
            print("?", _id, "invalid player", "\n", flush=True)

    @abstractmethod
    def boardsize(self, size: int) -> bool:
        return True

    @abstractmethod
    def clear_board(self):
        pass

    @abstractmethod
    def play(self, player: GoPlayer, pos: GoPoint) -> bool:
        return True

    @abstractmethod
    def genmove(self, player: GoPlayer) -> Union[GoPoint, str]:
        """
        return pass or resign
        """
        return "pass"

    def do_command(self, line: str):
        parsed = self.CMD.match(line)
        if parsed is not None:
            cmd = "do_" + parsed["command"]
            _id = parsed["id"]
            _id = "" if _id is None else _id
            if hasattr(self, cmd):
                func = getattr(self, cmd)
                return func(_id, parsed["args"])
            else:
                print("?", _id, "unknown command", "\n", flush=True)
        else:
            print("?", "unknown command", "\n", flush=True)

    def mainloop(self):
        while True:
            self.do_command(input())
