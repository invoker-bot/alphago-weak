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


def train_iter(epochs=100, steps=1, board_size=19, komi=6.5):
    bot = GTPAlphaGoWeakV0(board_size, komi)
    for epoch in range(epochs):
        X, P, V = bot.self_play(steps)
        dataset = tf.data.Dataset.from_tensor_slices((X, (P, V)))
        loss, acc = bot.model.fit_step_from_dataset(dataset)
        bot.model.save()
        # loss, accuracy = history.history["loss"][0], history.history["PolicyOutput_accuracy"][0]
        print(f"epoch {epoch}: Total Loss: {loss:.04f}, Policy Accuracy: {acc:.3%}")


if __name__ == '__main__':
    train_iter(board_size=5, epochs=1)
    # bot.model.save()
    # cpus = cpu_count() // 2
    # total = len(dataset)
    # model = AlphaGoWeak()
    # dataset, length = model.dataset_from_preprocess()
    # model.fit_from_dataset(dataset, length)
