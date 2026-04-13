from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


class Cell(IntEnum):
    EMPTY = -1
    BLACK = 0
    WHITE = 1


class DiskColor(IntEnum):
    BLACK = 0
    WHITE = 1

    def opponent(self) -> DiskColor:
        return DiskColor.WHITE if self == DiskColor.BLACK else DiskColor.Black


# 8x8 ndarray of Cells
BoardState: TypeAlias = NDArray[np.int8]
# board, disk color to play
State: TypeAlias = tuple[BoardState, DiskColor]
# row, col, disk color to play
Action: TypeAlias = tuple[int, int, DiskColor]
