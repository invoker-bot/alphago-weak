# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 3/12/2021
@Version : 1.0
"""

import sys
import re
import string
from abc import abstractmethod
from typing import *

from ..basic import *

__all__ = ["GTPClient"]


class GTPClient:
    __version__ = "1.0"
    name = "none"
    CMD = re.compile(r"^\s* (?P<id>\d+)? \s* (?P<command>\w*) \s* (?P<args>.*)", re.VERBOSE)
    MOVE = re.compile(r"^\s*(?P<x>[abcdefghjklmnopqrst])(?P<y>\d{1,2})", re.VERBOSE | re.IGNORECASE)
    COLOR = re.compile(r"^\s* (?P<color>w|b|white|black) \s* (?P<move>\w+)?", re.VERBOSE | re.IGNORECASE)

    COORDINATE = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
    COORDINATE_R = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "J": 8, "K": 9, "L": 10, "M": 11,
                    "N": 12, "O": 13, "P": 14, "Q": 15, "R": 16, "S": 17, "T": 18}

    @classmethod
    def gtp_map(cls):
        _dict = {_cls.name: _cls for _cls in cls.__subclasses__()}
        for v in cls.__subclasses__():
            _dict.update(v.gtp_map())
        return _dict

    def __init__(self):
        self.komi = 6.5

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
            if 0 < size <= 19 and self._do_boardsize(size):
                print("=", _id, "\n", flush=True)
            else:
                print("?", _id, "unacceptable size", "\n", flush=True)
        except ValueError:
            print("?", _id, "boardsize not an integer", "\n", flush=True)

    def do_komi(self, _id: str = "", args: str = ""):
        try:
            self.komi = float(args)
            print("=", _id, "\n", flush=True)
        except ValueError:
            print("?", _id, "komi not a float", "\n", flush=True)

    def do_clear_board(self, _id: str = "", args: str = ""):
        self._do_clear_board()
        print("=", _id, "\n", flush=True)

    def do_play(self, _id: str = "", args: str = ""):
        parsed = self.COLOR.match(args)
        if parsed and parsed["move"]:
            color = parsed["color"].lower()
            action = parsed["move"].strip().lower()
            if action == "pass" or action == "resign":
                print("=", _id, "\n", flush=True)
            else:
                move_parsed = self.MOVE.match(action)
                x = self.COORDINATE_R[move_parsed["x"].upper()]
                y = int(move_parsed["y"]) - 1
                if self._do_play(GoPlayer.to_player(color), (x, y)):
                    print("=", _id, "\n", flush=True)
                else:
                    print("?", _id, "illegal move", "\n", flush=True)

        else:
            print("?“,”invalid color or coordinate", "\n", flush=True)

    def do_genmove(self, _id: str = "", args: str = ""):
        parsed = self.COLOR.match(args)
        if parsed:
            player = GoPlayer.to_player(parsed["color"].lower())
            result = self._do_genmove(player)
            if isinstance(result, tuple):
                self._do_play(player, result)
                result = self.COORDINATE[result[0]] + str(result[1] + 1)
            print("=", _id, result, "\n", flush=True)
        else:
            print("?", _id, "invalid color", "\n", flush=True)

    @abstractmethod
    def _do_boardsize(self, size: int) -> bool:
        return True

    @abstractmethod
    def _do_clear_board(self) -> NoReturn:
        pass

    @abstractmethod
    def _do_play(self, color: GoPlayer, pos: GoPoint) -> bool:
        return True

    @abstractmethod
    def _do_genmove(self, color: GoPlayer) -> Union[GoPoint, str]:
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
