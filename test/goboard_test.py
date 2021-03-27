from unittest import TestCase, main, skip, expectedFailure
import sys

if ".." not in sys.path:
    sys.path.append("..")
from old.goboard_v0 import GoBoard as GoBoardV0
from go_board import *
from goboard_check_v0 import *
from goboard_check_v1 import *
import random


# import __main__
# __main__.SgfFile = SgfFile

@skip("deprecated")
class TestGoBoardV0(TestCase):

    def test_v0(self):
        check_v0(self, GoBoardV0)


class TestGoBoard(TestCase):

    def test_v0(self):
        check_v0(self, GoBoard)


if __name__ == '__main__':
    main()
