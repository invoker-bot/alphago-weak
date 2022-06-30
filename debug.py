#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/27
@Version : 1.0
"""

from dlgo.gotypes import *
from dlgo.goboard import Board
from alphago_weak.board import *
from alphago_weak.dataset import GameData

if __name__ == '__main__':
    b = Board(19, 19)
    b.place_stone(Player.black, Point(3, 3))
    print(b.get(Point(3, 3)))
    print(b.get(Point(0, 0)))
    data = GameData.from_sgf("test/sgf/2019-04-01-1.sgf")
    data.to_sgf("test.sgf")
