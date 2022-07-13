# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/27
@Version : 1.1
"""

import tqdm
from os import cpu_count
from typing import *
from concurrent.futures import *

__all__ = ["do_works"]

T = TypeVar('T')
V = TypeVar('V')

work_max_num = 10000


def do_works(func: Callable[[T], V], works: List[T], desc: str, unit: str = "it", cpu=True) -> List[V]:
    R = []
    unit_scale = True if len(works) > work_max_num else False
    with tqdm.tqdm(total=len(works), desc=desc, unit=unit, unit_scale=unit_scale) as bar:
        PoolExecutor = ProcessPoolExecutor if cpu else ThreadPoolExecutor
        cpus = cpu_count() // 2 if cpu else cpu_count()
        with PoolExecutor(max_workers=cpus) as executor:
            while len(works) > 0:
                processing_works = works[:work_max_num]
                works = works[work_max_num:]
                results = executor.map(func, processing_works)
                for r in results:
                    R.append(r)
                    bar.update(1)
    return R
