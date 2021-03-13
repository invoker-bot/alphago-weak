# -*- coding: utf-8 -*-

from typing import *
import numpy as np
from os import path
import time
from abc import ABCMeta, abstractmethod
from go_cache import *

__all__ = ["AlphaGoBase"]


class AlphaGoBase(metaclass=ABCMeta):
    def __init__(self, mtype: str, size: int = 19, dtype=np.float16) -> NoReturn:
        self.size = size
        self.dtype = dtype
        self.mtype = mtype

    def log_dir(self, name: str = time.strftime("%Y_%m_%d-%H_%M_%S")):
        return path.join(self.root(), "logs", name)

    def root(self):
        return path.join(get_cache_dir(), "models", self.mtype)

    @abstractmethod
    def fit(self, sample: str, epochs=100, batch_size=16):
        pass
