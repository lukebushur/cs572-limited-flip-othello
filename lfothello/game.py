from typing import Optional
from abc import ABC, abstractmethod

import numpy as np

from .types import Cell, DiskColor, BoardState, State, Action


BOARD_SIZE = 8


class Player(ABC):
    def __init__(self, color: DiskColor) -> None:
        self.color = color

    @abstractmethod
    def get_next_move(self, state: State, actions: list[Action]) -> Action:
        pass


class CLIPlayer(Player):
    def __init__(self, color: DiskColor) -> None:
        super().__init__(color)

    def get_next_move(self, state: State, actions: list[Action]) -> Action:
        return (0, 0, self.color)


class Board:
    def __init__(self, initial: Optional[BoardState]) -> None:
        if initial:
            self.state: State = (initial, DiskColor.BLACK)
        else:
            board = np.full((BOARD_SIZE, BOARD_SIZE), Cell.EMPTY)
            board[3, 3] = Cell.WHITE
            board[3, 4] = Cell.BLACK
            board[4, 3] = Cell.BLACK
            board[4, 4] = Cell.WHITE

            self.state: State = (board, DiskColor.BLACK)

    def get_state(self) -> State:
        return self.state

    def _get_opponent(self) -> DiskColor:
        if self.state[1] == DiskColor.BLACK:
            return DiskColor.WHITE
        else:
            return DiskColor.BLACK

    def get_actions(self) -> list[Action]:
        board, _ = self.state
        opp = self._get_opponent()
        opp_coords = np.argwhere(board == opp)
        print(opp_coords)
        offsets = {
            "top": (-1, 0),
            "right": (0, 1),
            "bottom": (1, 0),
            "left": (0, -1),
        }
        candidate_moves: set[tuple[int, int]] = set()
        for coord in opp_coords:
            for offset in offsets.values():
                move = coord + offset
                i, j = int(move[0]), int(move[1])
                if board[i, j] == Cell.EMPTY:
                    candidate_moves.add((i, j))

        # TODO
        # for move in candidate_moves:
        return []

    def _get_cell_display(self, cell_value: Cell) -> str:
        match cell_value:
            case Cell.EMPTY:
                return "."
            case Cell.BLACK:
                return "O"
            case Cell.WHITE:
                return "@"

    def display_state(self) -> None:
        print("+--------+")
        for i in range(BOARD_SIZE):
            print("|", end="")
            for j in range(BOARD_SIZE):
                board = self.state[0]
                cell = self._get_cell_display(board[i, j])
                print(cell, end="")
            print("|")
        print("+--------+")
