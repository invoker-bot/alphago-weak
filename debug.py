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
import os
import numpy as np
from os import cpu_count, path
from concurrent.futures import *
from alphago_weak.board import *
from alphago_weak.dataset import *
from alphago_weak.model.alpha_go_weak import *
from alphago_weak.gtp.gtp_alphago_weak import *

P = GoPoint
B, W = GoPlayer.black, GoPlayer.white


def train_iter(num=100, board_size=19, steps=16):
    for _ in range(num):
        bot = GTPAlphaGoWeakV0()
        bot.boardsize(board_size)
        bot.self_play(board_size, steps)
        cached_dir = bot.model.cache_dir
        X = np.load(path.join(cached_dir, "self_play.input.npy"))
        P = np.load(path.join(cached_dir, "self_play.policy_output.npy"))
        V = np.load(path.join(cached_dir, "self_play.value_output.npy"))
        history = bot.model.model.fit(X, (P, V), batch_size=32, epochs=1)
        bot.model.save()
        # loss, accuracy = history.history["loss"][0], history.history["PolicyOutput_accuracy"][0]
        # print("loss:", loss, "accuracy:", accuracy)


if __name__ == '__main__':
    train_iter(board_size=9)

    # bot.model.save()
    # cpus = cpu_count() // 2
    # total = len(dataset)
    # model = AlphaGoWeak()
    # dataset, length = model.dataset_from_preprocess()
    # model.fit_from_dataset(dataset, length)
