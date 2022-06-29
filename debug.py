#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/27
@Version : 1.0
"""

from alphago_weak.dataset import GameArchive
from alphago_weak.model.alpha_go_weak import AlphaGoWeakV0

if __name__ == '__main__':
    archive = GameArchive()
    model = AlphaGoWeakV0("alpha_go_weak_v0")
    model.fit(archive)

