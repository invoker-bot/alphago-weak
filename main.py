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

from src import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGo 1.0")
    parser.add_argument("--datadir", default=None, type=str, dest="datadir", help="the directory to cache data")
    parser.add_argument("-v", "--version", action="version", version="AlphaGo 1.0")
    sub_parser = parser.add_subparsers(dest="command", help="commands")

    download_map = GameArchive.archive_map()

    download_parser = sub_parser.add_parser("dataset", help="dataset go data from the Internet")
    download_parser.add_argument("--type", choices=download_map.keys(), required=True, help="type of the website")
    download_parser.add_argument("-f", "--force", action="store_true", default=False, dest="force",
                                 help="force to dataset")

    train_map = AlphaGoBase.alphago_map()
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

    play_parser = sub_parser.add_parser("play", help="play the game of go by GTP")
    play_map = GTPClient.gtp_map()
    play_parser.add_argument("--mode", choices=play_map.keys(), required=True, help="mode of the bot")
    args = parser.parse_args()
    if args.command == "dataset":
        set_cache_dir(args.datadir)
        ar = download_map[args.type]()
        ar.download(args.force)
    elif args.command == "play":
        bot = play_map[args.mode]()
        bot.mainloop()
    elif args.command == "train":
        set_cache_dir(args.datadir)
        al: AlphaGoBase = train_map[args.type](args.size)
        weights_file = args.weight if args.weight.endswith(".h5") else args.weight + ".h5"
        if args.init:
            al.init_fit(args.sample, weights_file, force=args.force)
