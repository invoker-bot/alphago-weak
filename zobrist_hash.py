import numpy as np
from gotypes import GoPoint
SIZE_MAX = 19

np.random.seed(0)

INT_TYPE = np.int64
INT_MAX = 0x7fff_ffff_ffff_ffff

ZOBRIST_HASHMAP = np.random.randint(-INT_MAX, INT_MAX, size=(3, SIZE_MAX, SIZE_MAX), dtype=INT_TYPE)


def get_hash(color: int, point: GoPoint) -> INT_TYPE:
    return ZOBRIST_HASHMAP.item((color, point[0], point[1]))


def resize(size: int):
    global SIZE_MAX, ZOBRIST_HASHMAP
    if size > SIZE_MAX:
        _NEW = np.random.randint(-INT_MAX, INT_MAX, size=(3, size, size), dtype=INT_TYPE)
        _NEW[:, 0:SIZE_MAX, 0:SIZE_MAX] = ZOBRIST_HASHMAP
        SIZE_MAX = size
        ZOBRIST_HASHMAP = _NEW

