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
from os import cpu_count, path
from concurrent.futures import *
from alphago_weak.board import *
from alphago_weak.dataset import *
from alphago_weak.model.alpha_go_weak import *


def np_file(idx: int, suffix: str):
    return path.join(model.cache_dir, f"{idx}.{suffix}.npy")


def rename(target: int):
    for j in range(13786, target, -1):
        if path.exists(np_file(j, "input")):
            os.rename(np_file(j, "input"), np_file(target, "input"))
            os.rename(np_file(j, "policy_output"), np_file(target, "policy_output"))
            os.rename(np_file(j, "value_output"), np_file(target, "value_output"))
            return True
    return False

if __name__ == '__main__':

    # bar = tqdm.tqdm(total=len(dataset), desc="testing")
    # with ProcessPoolExecutor(max_workers=cpu_count()//2) as executor:
    #    for _ in executor.map(test, dataset):
    #        bar.update(1)
    # test(dataset[238])

    # cpus = cpu_count() // 2
    # total = len(dataset)
    model = AlphaGoWeak()
    dataset, length = model.dataset_from_preprocess()
    model.fit_from_dataset(dataset, length)

