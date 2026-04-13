from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


class Cell(IntEnum):
    """
    Represents the contents of a single board cell.

    Values are stored as integers for efficient use in NumPy arrays.

    Attributes
    ----------
    EMPTY : int
        Represents an empty cell (-1).
    BLACK : int
        Represents a black disk (0).
    WHITE : int
        Represents a white disk (1).
    """
    EMPTY = -1
    BLACK = 0
    WHITE = 1


class DiskColor(IntEnum):
    """
    Represents the player (disk color) whose turn it is.

    Attributes
    ----------
    BLACK : int
        Black player (0).
    WHITE : int
        White player (1).
    """
    BLACK = 0
    WHITE = 1

    def opponent(self) -> DiskColor:
        """
        Return the opposing player's color.

        Returns
        -------
        DiskColor
            The opposite color (BLACK -> WHITE, WHITE -> BLACK).
        """
        return DiskColor.WHITE if self == DiskColor.BLACK else DiskColor.BLACK


# 8x8 ndarray of Cells
BoardState: TypeAlias = NDArray[np.int8]
# current board config, disk color to play
State: TypeAlias = tuple[BoardState, DiskColor]
# row, col, disk color to play (which player is making the move)
Action: TypeAlias = tuple[int, int, DiskColor]
