#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/27
@Version : 1.0
"""
import tensorflow as tf
from alphago_weak.board import *
from alphago_weak.gtp.gtp_random_bot_mcts import *
from alphago_weak.gtp.gtp_alphago_weak import *

P = GoPoint
B, W = GoPlayer.black, GoPlayer.white


if __name__ == '__main__':
    bot = GTPAlphaGoWeakV0()
    print(bot.genmove(GoPlayer.black))
    # bot.model.save()
    # cpus = cpu_count() // 2
    # total = len(dataset)
    # model = AlphaGoWeak()
    # dataset, length = model.dataset_from_preprocess()
    # model.fit_from_dataset(dataset, length)
