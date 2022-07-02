#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/27
@Version : 1.0
"""
from alphago_weak.board import *
from alphago_weak.gtp.gtp_random_bot_mcts import *
from alphago_weak.dataset import *
from alphago_weak.model.alpha_go_weak import AlphaGoWeakV0


def average(archive, num):
    it = iter(archive)
    length = [len(next(it).sequence) for _ in range(num)]
    return sum(length) / len(length)


if __name__ == '__main__':
    # a = GTPRandomBotMCTS()
    # a.boardsize(7)
    # print(a.genmove(GoPlayer.black))
    dataset = GameArchive()
    model = AlphaGoWeakV0()
    model.fit(dataset)
