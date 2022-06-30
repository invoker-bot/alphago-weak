# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/30
@Version : 1.0
"""


class Factory(object):

    _OBJ_DICT = {}

    @classmethod
    def register(cls, name: str, args: tuple = None, kwargs: dict = None):
        info = {
            "args": tuple() if args is None else args,
            "kwargs": {} if kwargs is None else kwargs,
        }

        def _wrapper(constructor):
            info["constructor"] = constructor
            cls._OBJ_DICT[name] = info
            return constructor

        return _wrapper

    @classmethod
    def create(cls, name: str):
        info = cls._OBJ_DICT[name]
        return info["constructor"](*info["args"], **info["kwargs"])

    @classmethod
    def names(cls):
        return list(cls._OBJ_DICT.keys())
