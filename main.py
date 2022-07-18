#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2022/7/18
@Version : 1.0
"""
import shutil
import argparse
import tensorflow as tf
from alphago_weak.dataset import *
from alphago_weak.gtp import *
from alphago_weak.model.alpha_go_weak import *


def download(args):
    archive = GameArchive.FACTORY_DICT[args.type](args.root)
    archive.download(args.force)


def clean(args):
    shutil.rmtree(args.root)


def train(args):
    archive = GameArchive(args.root)
    if args.init > 0:
        AlphaGoWeak.preprocess_archive(archive, args.init)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        dataset, steps = AlphaGoWeak.dataset_from_preprocess(archive)
        model = AlphaGoWeak(root=args.root)
    model.fit_from_dataset(dataset, steps, args.epochs, args.batch_size * strategy.num_replicas_in_sync)


def evaluate(args):
    black_v = GTPClientBase.evaluate(args.black, args.white, args.board_size, args.count, args.komi, args.output, args.multiprocessing)
    print("the winning rate of black bot (%s): %0.3f" % (args.black, black_v))
    print("the winning rate of white bot (%s): %0.3f" % (args.white, 1 - black_v))


def play(args):
    bot = GTPClientBase.FACTORY_DICT[args.type](None)
    bot.cmdloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An experimental AlphaGo similar program")
    parser.add_argument("-r", "--root", default=".data", help="root directory for the datasets to cache")
    parser.add_argument("-v", "--version", action="version", version="AlphaGo Weak v1.0")
    sub_parser = parser.add_subparsers(title="commands", dest="command", required=True)

    download_parser = sub_parser.add_parser("download", help="download the datasets from the Internet, currently only supports board size with 19")
    download_types = list(GameArchive.FACTORY_DICT.keys())
    download_parser.add_argument("-t", "--type", choices=download_types, default=download_types[0], help="the type of the website to fetch dataset")
    download_parser.add_argument("-f", "--force", action="store_true", default=False, dest="force",
                                 help="whether to download the datasets forcely")
    download_parser.set_defaults(func=download)

    clean_parser = sub_parser.add_parser("clean", help="clean cached the datasets, useful to save storage space")
    clean_parser.set_defaults(func=clean)
    
    train_parser = sub_parser.add_parser("train", help="train AlphaGo Weak from datasets downloaded, currently only supports board size with 19")
    train_parser.add_argument("--init", type=int, default=0, help="initialize with the specified size of dataset for training")
    train_parser.add_argument("--batch_size", type=int, default=512, help="training batch size")
    train_parser.add_argument("--epochs", type=int, default=100, help="max training epochs")
    


    train_parser.set_defaults(func=train)

    evaluate_parser = sub_parser.add_parser("evaluate", help="evaluate the level of two bots from matches")
    evaluate_parser.add_argument("-b", "--black", choices=GTPClientBase.FACTORY_DICT.keys(), required=True, help="the type of the black bot")
    evaluate_parser.add_argument("-w", "--white", choices=GTPClientBase.FACTORY_DICT.keys(), required=True, help="the type of the white bot")
    evaluate_parser.add_argument("--board_size", type=int, default=19, help="the size of the board")
    evaluate_parser.add_argument("--komi", type=float, default=6.5, help="the extra scores given to white when scoring")
    evaluate_parser.add_argument("-c", "--count", type=int, default=100, help="the counts of matches")
    evaluate_parser.add_argument("-o", "--output", default=None, help="output directory for antagonistic process (SGF format), useful for researching or visualization")
    evaluate_parser.add_argument("--multiprocessing", action="store_true", default=False, help="whether use multiprocessing, it speeds up sometimes but when including GPU process it leads breaking")
    evaluate_parser.set_defaults(func=evaluate)

    play_parser = sub_parser.add_parser("play", help="play by GTP protocol, useful for connecting with a GO GUI application")
    play_parser.add_argument("-t", "--type", choices=GTPClientBase.FACTORY_DICT.keys(), required=True, help="the type of the bot")
    play_parser.set_defaults(func=play)

    args = parser.parse_args()
    args.func(args)

