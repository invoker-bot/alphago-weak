#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from os import path
from glob import iglob
from alphago_weak.board.board import GoBoard
from alphago_weak.utils.board_check import *


class TestGoBoard(unittest.TestCase):

    def test_basic(self):
        board_check_basic(self, GoBoard)

    def test_sgf(self):
        for sgf_name in iglob(path.join(path.dirname(__file__), "sgf/*.sgf")):
            board_check_sgf(self, sgf_name, GoBoard)


if __name__ == '__main__':
    unittest.main()
