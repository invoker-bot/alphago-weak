# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/6/27
@Version : 1.1
"""

from os import cpu_count
from typing import *
from concurrent.futures import *
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

__all__ = ["do_works_experimental", "do_cpu_intensive_works"]

T = TypeVar('T')
V = TypeVar('V')

work_max_num = 10000


def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def do_works_experimental(func: Callable[[T], V], works: List[T], desc: str, unit: str = "it", cpu=True) -> List[V]:
    R = []
    with _progress() as progress:
        task = progress.add_task(f"{desc} ({unit})", total=len(works))
        PoolExecutor = ProcessPoolExecutor if cpu else ThreadPoolExecutor
        cpus = max(1, (cpu_count() or 1) // 2) if cpu else max(1, cpu_count() or 1)
        with PoolExecutor(max_workers=cpus) as executor:
            while len(works) > 0:
                processing_works = works[:work_max_num]
                works = works[work_max_num:]
                results = executor.map(func, processing_works)
                for r in results:
                    R.append(r)
                    progress.advance(task)
    return R


def do_cpu_intensive_works(func: Callable[[T], V], works: Iterable[T], total: int = None, desc: str = None, use_multiprocessing=False, max_workers: int = None) -> Iterator[V]:
    description = desc or "Working..."
    if not use_multiprocessing:
        results = map(func, works)
        if total is None:
            yield from results
        else:
            with _progress() as progress:
                task = progress.add_task(description, total=total)
                for result in results:
                    progress.advance(task)
                    yield result
    else:
        with ProcessPoolExecutor(max_workers) as executor:
            results = executor.map(func, works)
            if total is None:
                yield from results
            else:
                with _progress() as progress:
                    task = progress.add_task(description, total=total)
                    for result in results:
                        progress.advance(task)
                        yield result
