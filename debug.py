#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/27
@Version : 1.0
"""
import tqdm
from os import cpu_count
from concurrent.futures import *
from alphago_weak.board import *
from alphago_weak.dataset import *
from alphago_weak.model.alpha_go_weak import *


def average(archive, num):
    it = iter(archive)
    length = [len(next(it).sequence) for _ in range(num)]
    return sum(length) / len(length)


def test(data: GameData):
    b = GoBoardAlpha(data.size)
    b.setup_stones(*data.setup_stones)
    for steps, (player, pos) in enumerate(data.sequence):
        if pos is not None:
            pos = GoPoint(*pos)
        try:
            b.play(player, pos)
        except GoIllegalActionError:
            print("step:", steps)
            raise



if __name__ == '__main__':
    # a = GTPRandomBotMCTS()
    # a.boardsize(7)
    # print(a.genmove(GoPlayer.black))

    dataset = GameArchive()
    #bar = tqdm.tqdm(total=len(dataset), desc="testing")
    #with ProcessPoolExecutor(max_workers=cpu_count()//2) as executor:
    #    for _ in executor.map(test, dataset):
    #        bar.update(1)
    # test(dataset[238])

    # cpus = cpu_count() // 2
    # total = len(dataset)
    model = AlphaGoWeak()
    model.fit_from_archive(dataset)
