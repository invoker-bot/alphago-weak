#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from alphago_weak.board.board import GoBoard
from alphago_weak.utils.board_check import board_check_v0


class TestGoBoard(unittest.TestCase):

    def test_v0(self):
        board_check_v0(self, GoBoard)


if __name__ == '__main__':
    unittest.main()
