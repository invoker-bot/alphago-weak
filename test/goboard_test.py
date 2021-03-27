#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import sys

if ".." not in sys.path:
    sys.path.append("..")
from goboard_check_v0 import *


# import __main__
# __main__.SgfFile = SgfFile


class TestGoBoard(TestCase):

    def test_v0(self):
        check_board_v0(self, GoBoard)


if __name__ == '__main__':
    unittest.main()
