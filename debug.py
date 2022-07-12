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
import math
import timeit
import tensorflow as tf
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


def game_data_generator(archive: GameArchive):
    for data in archive:
        b = GoBoard(data.size)
        b.setup_stones(*data.setup_stones)
        for steps, (player, pos) in enumerate(data.sequence):
            if pos is not None:
                pos = GoPoint(*pos)
            x = math.exp(- steps / (data.size * data.size))
            decay = min(2 * (1 - x) / (1 + x), 1.0)
            yield AlphaGoWeakV0.encode_input(player, b), (
                AlphaGoWeakV0.encode_policy_output(b.size, pos),
                AlphaGoWeakV0.encode_value_output(player, data.winner, decay))
            b.play(player, pos)


def load(num=32):
    data = tf.data.Dataset.from_generator(lambda: game_data_generator(GameArchive()),
                                   output_signature=(
                                       tf.TensorSpec((11, 19, 19)),
                                       (tf.TensorSpec((19 * 19 + 1,)),
                                        tf.TensorSpec(())))).take(num)
    return list(data)


def load2(num=32):


    data = tf.data.Dataset.range(4).interleave(lambda _:tf.data.Dataset.from_generator(lambda: game_data_generator(GameArchive()),
                                          output_signature=(
                                              tf.TensorSpec((11, 19, 19)),
                                              (tf.TensorSpec((19 * 19 + 1,)),
                                               tf.TensorSpec(())))), num_parallel_calls=4).take(num)
    return list(data)

if __name__ == '__main__':
    # a = GTPRandomBotMCTS()
    # a.boardsize(7)
    # print(a.genmove(GoPlayer.black))

    dataset = GameArchive()
    print(timeit.timeit("load(1024)", "from __main__ import load",number=10) / 1024)
    print(timeit.timeit("load2(1024)", "from __main__ import load2", number=10) / 1024)
    # bar = tqdm.tqdm(total=len(dataset), desc="testing")
    # with ProcessPoolExecutor(max_workers=cpu_count()//2) as executor:
    #    for _ in executor.map(test, dataset):
    #        bar.update(1)
    # test(dataset[238])

    # cpus = cpu_count() // 2
    # total = len(dataset)
    # model = AlphaGoWeak()
    # model.preprocess_from_archive(dataset)
