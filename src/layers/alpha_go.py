# -*- coding: utf-8 -*-

from typing import *
import numpy as np
from os import path
import time
from abc import ABCMeta, abstractmethod

from ..basic import *

__all__ = ["AlphaGoBase"]


class AlphaGoBase(metaclass=ABCMeta):
    name = "alphago"

    @classmethod
    def alphago_map(cls):
        _dict = {_cls.name: _cls for _cls in cls.__subclasses__()}
        for v in cls.__subclasses__():
            _dict.update(v.alphago_map())
        return _dict

    def __init__(self, size: int = 19, dtype=np.float16) -> NoReturn:
        self.size = size
        self.dtype = dtype

    def log_dir(self, name: str = time.strftime("%Y_%m_%d-%H_%M_%S")):
        return path.join(self.root(), "logs", name)

    def root(self):
        return path.join(get_cache_dir(), "models", self.name)

    def weights_file(self, file: str = "default.h5"):
        return path.join(path.dirname(path.realpath(__file__)), "../../weights", self.name, str(self.size), file)

    @abstractmethod
    def init_fit(self, sample: str, weights_file: str, epochs=100, batch_size=16, force=False):
        pass
