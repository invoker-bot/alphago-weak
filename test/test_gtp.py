#!/usr/bin/env python
# -*- coding: utf-8 -*-

from alphago_weak.board import GoPlayer, GoPoint
from alphago_weak.gtp.gtp_random_bot import GTPRandomBot


def test_gtp_play_rejects_occupied_point():
    bot = GTPRandomBot(5)
    pos = GoPoint(0, 0)

    assert bot.play(GoPlayer.black, pos) is True
    assert bot.play(GoPlayer.white, pos) is False
    assert bot.board[pos] == GoPlayer.black


def test_gtp_play_rejects_out_of_bounds_point():
    bot = GTPRandomBot(5)

    assert bot.play(GoPlayer.black, GoPoint(0, -1)) is False
    assert bot.play(GoPlayer.black, GoPoint(5, 0)) is False
