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
    parser.add_argument("--datadir", default=".data", type=str, dest="datadir", help="the directory to cache data")
    parser.add_argument("-v", "--version", action="version", version="AlphaGo 1.0")
    sub_parser = parser.add_subparsers(dest="command", help="commands")

    download_parser = sub_parser.add_parser("download", help="download go data from the Internet")
    download_parser.add_argument("-f", "--force", action="store_true", default=False, dest="force",
                                 help="force to download")

    train_parser = sub_parser.add_parser("train", help="train AlphaGo Weak")
    train_parser.add_argument("--init", action="store_true", default=False, dest="init",
                              help="train AlphaGo Weak initially")
    train_parser.add_argument("-w", "--weight", action="store", default="default", dest="weight", help="weights file")
    train_parser.add_argument("-s", "--size", type=int, default=19, dest="size", help="size of the board")

    play_parser = sub_parser.add_parser("play", help="play the game of go by GTP")
    play_map = GTPClient.gtp_map()
    play_parser.add_argument("--mode", choices=play_map.keys(), required=True, help="mode of the bot")
    args = parser.parse_args()
    if args.command == "download":
        set_cache_dir(args.datadir)
        ar = UGoArchive()
        ar.download(args.force)
    elif args.command == "play":
        bot = play_map[args.mode]()
        bot.mainloop()
