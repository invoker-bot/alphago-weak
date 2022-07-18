# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2022/6/29
@Version : 1.1
"""

import re
import string
from os import path, makedirs
from cmd import Cmd
from importlib import import_module
from functools import partial
from abc import *
from typing import *

from ..board import *
from ..dataset import GameData
from ..utils.multi_works import do_cpu_intensive_works

__all__ = ["GTPClientBase"]

PKG = "alphago_weak.gtp"


def import_class_func(module: str, class_: str) -> Callable[[int, float, dict], "GTPClientBase"]:
    return lambda size, komi, kwargs: getattr(import_module(module, "alphago_weak.gtp"), class_)(size, komi, **kwargs)


class GTPClientBase(Cmd, metaclass=ABCMeta):
    __version__ = "1.0"
    prompt = ""
    board: GoBoardBase
    name: str = None
    FACTORY_DICT = {
        "random_bot": import_class_func(".gtp_random_bot", "GTPRandomBot"),
        "random_bot_mcts": import_class_func(".gtp_random_bot_mcts", "GTPRandomBotMCTS"),
        "alphago_weak_v0": import_class_func(".gtp_alphago_weak", "GTPAlphaGoWeakV0"),
    }

    PRECMD_PAT = re.compile(r"^\s* (?P<id>\d+)? \s* (?P<command>.*)", re.VERBOSE)
    MOVE = re.compile(r"^\s*(?P<x>[a-hj-z])(?P<y>\d{1,2})", re.VERBOSE | re.IGNORECASE)
    PLAY_PAT = re.compile(r"^\s* (?P<player>w|b|white|black) \s* (?P<move>\w+)?", re.VERBOSE | re.IGNORECASE)

    COORDINATE = tuple("ABCDEFGHJKLMNOPQRSTUVWXYZ")
    COORDINATE_R = {coor: idx for idx, coor in enumerate(COORDINATE)}

    def __init__(self, size=19, komi=6.5):
        super().__init__()
        self._size = size
        self.id = ""
        self.komi = komi

    @property
    def size(self) -> int:
        return self._size

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
    def _evaluate_one(cls, black: str, white: str, idx: int, board_size=19, komi=6.5, output: str = None):
        board = GoBoard(board_size)
        bots: Dict[GoPlayer, "GTPClientBase"] = {player: cls.FACTORY_DICT[bot_name](board_size, komi, {})
                                                 for player, bot_name in ((GoPlayer.black, black), (GoPlayer.white, white))}
        seq: List[Tuple[GoPlayer, GoPoint]] = []
        current_player = GoPlayer.black
        history_action = None
        while True:
            current_bot = bots[current_player]
            current_action = current_bot.genmove(current_player)
            if isinstance(current_action, GoPoint):
                board.play(current_player, current_action)
                for bot in bots.values():
                    bot.play(current_player, current_action)
                seq.append((current_player, current_action))
            elif current_action == "resign":
                break
            else:  # current_action == pass
                if history_action == "pass":
                    # set current play as a loser
                    current_player = GoPlayer.black if board.score(GoPlayer.black, komi) < 0 else GoPlayer.white
                    break
            history_action = current_action
            current_player = current_player.other
        winner = current_player.other
        if output is not None:
            game_data = GameData(board_size, winner, seq, komi)
            game_data.to_sgf(path.join(output, f"{idx}.sgf"))
        return board.score(GoPlayer.black, komi) > 0

    @classmethod
    def evaluate(cls, black: str, white: str, board_size=19, num=100, komi=6.5, output: str = None, use_multiprocessing=False):
        if output is not None:
            makedirs(output, exist_ok=True)
        black_count = sum(do_cpu_intensive_works(partial(cls._evaluate_one, black, white, board_size=board_size, komi=komi, output=output), range(num), total=num, desc="Evaluating...", use_multiprocessing=use_multiprocessing))
        return black_count / num

    def boardsize(self, size):
        self._size = size
        self.clear_board()
        return True

    def clear_board(self):
        self.board = self.board.__class__(self._size)

    def play(self, player: GoPlayer, pos: GoPoint):
        self.board[pos] = player
        return True

    @abstractmethod
    def genmove(self, player: GoPlayer) -> Union[GoPoint, str]:
        """
        can return "pass" or "resign"
        """
        ...
