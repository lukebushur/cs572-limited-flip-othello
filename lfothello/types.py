from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class Cell(IntEnum):
    EMPTY = -1
    BLACK = 0
    WHITE = 1


class DiskColor(IntEnum):
    BLACK = 0
    WHITE = 1


# 8x8 ndarray of Cells
type BoardState = NDArray[np.int_]
# board, disk color to play
type State = tuple[BoardState, DiskColor]
# row, col, disk color to play
type Action = tuple[int, int, DiskColor]
