# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/28
@Version : 1.0
"""

from abc import *
from os import path, makedirs


class ModelBase(metaclass=ABCMeta):
    def __init__(self, name: str, root: str = None):
        if root is None:
            root = path.join(path.dirname(path.realpath(__file__)), "../..", ".data")
        self.root = root
        self.name = name
        makedirs(self.root, exist_ok=True)
        makedirs(self.model_dir, exist_ok=True)

    @property
    def model_dir(self):
        return path.join(self.root, "model")

    @property
    def weights_path(self):
        return path.join(self.model_dir, self.name + ".h5")

    @property
    def logs_dir(self):
        return path.join(self.model_dir, "logs", self.name)
