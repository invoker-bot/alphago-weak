# -*- coding: utf-8 -*-
# (C) Invoker, Inc. 2020-2021
# All rights reserved
# Licensed under Simplified BSD License (see LICENSE)

"""
requests.api

This module contains xxx.
This module is designed to xxx.
"""

from typing import *

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import logging
import time
import random
import tqdm
import enum

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGo 1.0")
    parser.add_argument("--datadir", default=".data", type=str, dest="datadir", help="the directory to cache data")
    parser.add_argument("-v", "--version", action="version", version="AlphaGo 1.0")
    sub_parser = parser.add_subparsers(help="commands")
    download_parser = sub_parser.add_parser("download", help="download go data from Internet")

    args = parser.parse_args()
    print(args.datadir)
