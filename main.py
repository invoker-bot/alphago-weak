# -*- coding: utf-8 -*-
# (C) Invoker, Inc. 2020-2021
# All rights reserved
# Licensed under Simplified BSD License (see LICENSE)

"""
requests.api

This module contains xxx.
This module is designed to xxx.
"""

import argparse

from sympy import evaluate

from alphago_weak.dataset import *
from alphago_weak.gtp import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGo 1.0")
    parser.add_argument("--datadir", default=None, type=str, dest="datadir", help="the directory to cache data")
    parser.add_argument("-v", "--version", action="version", version="AlphaGo 1.0")
    sub_parser = parser.add_subparsers(dest="command", help="commands")

    download_parser = sub_parser.add_parser("download", help="download go dataset from the Internet")
    download_parser.add_argument("-r", "--root", default=None, help="root path of the dataset")
    download_parser.add_argument("-t", "--type", choices=GameArchive.FACTORY_DICT.keys(), required=True, help="type of the website")
    download_parser.add_argument("-f", "--force", action="store_true", default=False, dest="force",
                                 help="whether force to download dataset")

    """
    train_parser = sub_parser.add_parser("train", help="train AlphaGo Weak")
    train_parser.add_argument("--type", choices=train_map.keys(), required=True, help="type of the alphago")
    train_parser.add_argument("--init", action="store_true", default=False, dest="init",
                              help="train AlphaGo Weak initially")
    train_parser.add_argument("--sample", action="store", default="default_sample", dest="sample",
                              help="samples file of the AlphaGo")
    train_parser.add_argument("-w", "--weight", action="store", default="default", dest="weight",
                              help="weights file of the AlphaGo")
    train_parser.add_argument("-s", "--size", type=int, default=19, dest="size", help="size of the board")
    train_parser.add_argument("-f", "--force", action="store_true", default=False, dest="force",
                              help="force to train")
    """

    evaluate_parser = sub_parser.add_parser("evaluate", help="evaluate the level of two bots")
    evaluate_parser.add_argument("-b", "--black", choices=GTPClientBase.FACTORY_DICT.keys(), required=True, help="type of the black bot")
    evaluate_parser.add_argument("-w", "--white", choices=GTPClientBase.FACTORY_DICT.keys(), required=True, help="type of the white bot")
    evaluate_parser.add_argument("--board_size", type=int, default=19, help="go board size")
    evaluate_parser.add_argument("--komi", type=float, default=6.5, help="komi")
    evaluate_parser.add_argument("-c", "--count", type=int, default=100, help="evaluate counts")
    evaluate_parser.add_argument("-o", "--output", default=None, help="output directory for evaluate results")

    play_parser = sub_parser.add_parser("play", help="play the game of go by GTP")
    play_parser.add_argument("-t", "--type", choices=GTPClientBase.FACTORY_DICT.keys(), required=True, help="type of the bot")

    args = parser.parse_args()

    if args.command == "download":
        archive: GameArchive = GameArchive.FACTORY_DICT[args.type](args.root)
        archive.download(args.force)
    elif args.command == "evaluate":
        black_v = GTPClientBase.evaluate(args.black, args.white, board_size=args.board_size, num=args.count, komi=args.komi, output=args.output)
        print("the winning rate of black bot (%s): %0.3f" % (args.black, black_v))
        print("the winning rate of white bot (%s): %0.3f" % (args.white, 1 - black_v))

    elif args.command == "play":
        bot = GTPClientBase.FACTORY_DICT[args.type](None)
        bot.cmdloop()
