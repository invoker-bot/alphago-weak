#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/27
@Version : 1.0
"""
# from alphago_weak.board import *
# from alphago_weak.gtp.gtp_random_bot_mcts import *

from cmd import Cmd


class CmdObj(Cmd):

    prompt = ""

    def do_abs(self, arg):
        print("abc")

if __name__ == '__main__':
    # a = GTPRandomBotMCTS()
    # a.boardsize(7)
    # print(a.genmove(GoPlayer.black))
    cmd = CmdObj()
    cmd.cmdloop()