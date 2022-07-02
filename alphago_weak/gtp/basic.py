# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2022/6/29
@Version : 1.1
"""


import re
import tqdm
import string
from cmd import Cmd
from importlib import import_module
from abc import *
from typing import *

from ..board import *

__all__ = ["GTPClient"]

PKG = "alphago_weak.gtp"


class GTPClient(Cmd, metaclass=ABCMeta):
    __version__ = "1.0"
    prompt = ""
    name: str = None
    FACTORY_DICT = {
        "random_bot": lambda board: import_module(".gtp_random_bot", PKG).GTPRandomBot(board),
        "random_bot_mcts": lambda board: import_module(".gtp_random_bot_mcts", PKG).GTPRandomBotMCTS(board),
    }

    PRECMD_PAT = re.compile(r"^\s* (?P<id>\d+)? \s* (?P<command>.*)", re.VERBOSE)
    MOVE = re.compile(r"^\s*(?P<x>[a-hj-z])(?P<y>\d{1,2})", re.VERBOSE | re.IGNORECASE)
    PLAY_PAT = re.compile(r"^\s* (?P<player>w|b|white|black) \s* (?P<move>\w+)?", re.VERBOSE | re.IGNORECASE)

    COORDINATE = tuple("ABCDEFGHJKLMNOPQRSTUVWXYZ")
    COORDINATE_R = {coor: idx for idx, coor in enumerate(COORDINATE)}

    def __init__(self, board: GoBoardAlpha = None):
        super(Cmd, self).__init__()
        self.board = GoBoardAlpha() if board is None else board
        self.id = ""
        self.komi = 6.5
        self.config = {}

    def precmd(self, line: str) -> str:
        parsed = self.PRECMD_PAT.match(line)
        if parsed is not None:
            _id = parsed["id"]
            self.id = "" if _id is None else _id
            return parsed["command"]
        else:
            return ""

    def response(self, result: str = "\n", level="="):
        print(level, self.id, result, "\n", file=self.stdout, flush=True)

    def default(self, line):
        self.response("unknown command", "?")

    def do_protocol_version(self, args: str):
        self.response("2")

    def do_name(self, args: str):
        self.response(string.capwords(self.name, "_"))

    def do_version(self, args: str):
        self.response(self.__version__)

    def do_known_command(self, args: str):
        result = "true" if hasattr(self, "do_" + args.strip()) else "false"
        self.response(result)

    def do_list_commands(self, args: str):
        # cmds = [attr[3:] for attr in dir(self) if attr.startswith("do_")]
        self.response("\n".join(self.completenames("")))

    def do_quit(self, args: str):
        self.response()
        return True

    def do_boardsize(self, args: str):
        try:
            size = int(args)
            if 0 < size <= 25 and self.boardsize(size):
                self.response()
            else:
                self.response("unacceptable size", "?")
        except ValueError:
            self.response("boardsize is not an integer", "?")

    def do_komi(self, args: str):
        try:
            self.komi = float(args)
            self.response()
        except ValueError:
            self.response("komi not a float", "?")

    def do_clear_board(self, args: str):
        self.clear_board()
        self.response()

    def do_play(self, args: str):
        parsed = self.PLAY_PAT.match(args)
        if parsed and parsed["move"]:
            player = parsed["player"].lower()
            action = parsed["move"].strip().lower()
            if action == "pass" or action == "resign":
                self.response()
            else:
                move_parsed = self.MOVE.match(action)
                x = self.COORDINATE_R[move_parsed["x"].upper()]
                y = int(move_parsed["y"]) - 1
                if self.play(GoPlayer.to_player(player), GoPoint(x, y)):
                    self.response()
                else:
                    self.response("illegal move", "?")
        else:
            self.response("invalid player or coordinate", "?")

    def do_genmove(self, args: str):
        parsed = self.PLAY_PAT.match(args)
        if parsed:
            player = GoPlayer.to_player(parsed["player"].lower())
            result = self.genmove(player)
            if not isinstance(result, str):
                self.play(player, result)
                result = self.COORDINATE[result.x] + str(result.y + 1)
            self.response(result)
        else:
            self.response("invalid player", "?")

    @classmethod
    def evaluate(cls, black: str, white: str, board_size = 19, num: int = 100, komi = 6.5):
        black_count = 0
        for _ in tqdm.tqdm(range(num),total=num, unit="count", desc= "Simulating..."):
            board = GoBoardAlpha(board_size)
            black_bot: GTPClient = cls.FACTORY_DICT[black](board)
            black_bot.komi = komi
            white_bot: GTPClient = cls.FACTORY_DICT[white](board)
            white_bot.komi = komi
            while True:
                black_action = black_bot.genmove(GoPlayer.black)
                if isinstance(black_action, GoPoint):
                    board.play(GoPlayer.black, black_action)
                elif black_action == "resign":
                    break
                white_action = white_bot.genmove(GoPlayer.white)
                if isinstance(white_action, GoPoint):
                    board.play(GoPlayer.white, white_action)
                elif white_action == "resign":
                    black_count += 1
                    break
                if isinstance(black_action, str) and isinstance(white_action, str):
                    black_count += int(board.score(GoPlayer.black, komi) > 0)
                    break
        return black_count / num

    @abstractmethod
    def boardsize(self, size: int) -> bool:
        ...

    @abstractmethod
    def clear_board(self):
        ...

    @abstractmethod
    def play(self, player: GoPlayer, pos: GoPoint) -> bool:
        ...

    @abstractmethod
    def genmove(self, player: GoPlayer) -> Union[GoPoint, "str"]:
        """
        can return "pass" or "resign"
        """
        ...


