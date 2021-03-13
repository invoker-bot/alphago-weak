import numpy as np
from go_types import *

__all__ = ["get_hash"]

SIZE_MAX = 25

INT_TYPE = np.int64
INT_MAX = 0x7fff_ffff_ffff_ffff

np.random.seed(0xcfcfcfcf)
ZOBRIST_HASHMAP = np.random.randint(-INT_MAX, INT_MAX, size=(3, SIZE_MAX, SIZE_MAX), dtype=INT_TYPE)


def get_hash(color: int, point: GoPoint) -> INT_TYPE:
    return ZOBRIST_HASHMAP.item((color, point[0], point[1]))
